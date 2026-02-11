/**
 * Tests for neural network RNN/LSTM functionality
 */

import { describe, test, expect } from '@jest/globals';
import { LSTMNetwork, normalizeVector, cosineSimilarity, padSequence } from '../src/neural/rnn-core.js';
import { NEURAL_MODEL_CONFIGS } from '../src/neural/rnn-config.js';

describe('Neural Network Core', () => {
    test('should normalize a vector', () => {
        const vector = [1, 2, 3, 4, 5];
        const normalized = normalizeVector(vector);
        
        // Check that normalized vector has zero mean
        const mean = normalized.reduce((sum, val) => sum + val, 0) / normalized.length;
        expect(Math.abs(mean)).toBeLessThan(1e-10);
    });

    test('should calculate cosine similarity', () => {
        const a = [1, 0, 0];
        const b = [1, 0, 0];
        const c = [0, 1, 0];
        
        // Identical vectors should have similarity of 1
        expect(cosineSimilarity(a, b)).toBeCloseTo(1, 5);
        
        // Orthogonal vectors should have similarity of 0
        expect(cosineSimilarity(a, c)).toBeCloseTo(0, 5);
    });

    test('should pad sequence to target length', () => {
        const sequence = [[1, 2], [3, 4]];
        const targetLength = 5;
        const vectorSize = 2;
        
        const padded = padSequence(sequence, targetLength, vectorSize);
        
        expect(padded.length).toBe(targetLength);
        expect(padded[0]).toEqual([0, 0]); // Zero padding
        expect(padded[3]).toEqual([1, 2]); // Original data
        expect(padded[4]).toEqual([3, 4]);
    });

    test('should truncate sequence to target length', () => {
        const sequence = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]];
        const targetLength = 3;
        const vectorSize = 2;
        
        const truncated = padSequence(sequence, targetLength, vectorSize);
        
        expect(truncated.length).toBe(targetLength);
        // Should keep the most recent items
        expect(truncated[0]).toEqual([5, 6]);
        expect(truncated[1]).toEqual([7, 8]);
        expect(truncated[2]).toEqual([9, 10]);
    });
});

describe('LSTM Network', () => {
    test('should create LSTM network from config', () => {
        const network = new LSTMNetwork('character-evolution');
        
        expect(network.modelType).toBe('character-evolution');
        expect(network.config).toEqual(NEURAL_MODEL_CONFIGS['character-evolution']);
        expect(network.layers.length).toBe(network.config.numLayers);
    });

    test('should perform forward pass', () => {
        const network = new LSTMNetwork('context-relevance');
        const inputSize = network.config.inputSize;
        const input = Array(inputSize).fill(0).map(() => Math.random());
        
        const output = network.forward(input);
        
        expect(output.length).toBe(network.config.outputSize);
        expect(output.every(val => typeof val === 'number')).toBe(true);
    });

    test('should process sequence of inputs', () => {
        const network = new LSTMNetwork('narrative-flow');
        const inputSize = network.config.inputSize;
        const sequenceLength = 5;
        
        const sequence = Array(sequenceLength).fill(0).map(() =>
            Array(inputSize).fill(0).map(() => Math.random())
        );
        
        const outputs = network.forwardSequence(sequence);
        
        expect(outputs.length).toBe(sequenceLength);
        expect(outputs[0].length).toBe(network.config.outputSize);
    });

    test('should reset states', () => {
        const network = new LSTMNetwork('character-evolution');
        const inputSize = network.config.inputSize;
        
        // Process some inputs
        network.forward(Array(inputSize).fill(0.5));
        network.forward(Array(inputSize).fill(0.7));
        
        // Reset
        network.resetStates();
        
        // All states should be zero
        network.hiddenStates.forEach(state => {
            expect(state.every(val => val === 0)).toBe(true);
        });
        network.cellStates.forEach(state => {
            expect(state.every(val => val === 0)).toBe(true);
        });
    });

    test('should save and restore state', () => {
        const network = new LSTMNetwork('world-state-evolution');
        const inputSize = network.config.inputSize;
        
        // Process some inputs
        network.forward(Array(inputSize).fill(0.5));
        const savedState = network.getState();
        
        // Process more inputs
        network.forward(Array(inputSize).fill(0.7));
        
        // Restore saved state
        network.setState(savedState);
        
        // States should match saved state
        expect(network.hiddenStates).toEqual(savedState.hiddenStates);
        expect(network.cellStates).toEqual(savedState.cellStates);
    });
});
