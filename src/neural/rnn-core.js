/**
 * Core RNN/LSTM implementation for SillyTavern neural context system.
 * Provides adaptive world building and character evolution capabilities.
 */

import { getNeuralModelConfig } from './rnn-config.js';

/**
 * Simple LSTM cell implementation for JavaScript
 * This is a lightweight implementation that works with the existing infrastructure
 */
class LSTMCell {
    /**
     * @param {number} inputSize - Size of input vector
     * @param {number} hiddenSize - Size of hidden state
     */
    constructor(inputSize, hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        // Initialize weights (in production, these would be loaded from trained models)
        this.weights = this._initializeWeights();
    }

    /**
     * Initialize LSTM weights
     * @private
     */
    _initializeWeights() {
        const weights = {
            // Input gate
            Wii: this._randomMatrix(this.hiddenSize, this.inputSize),
            Whi: this._randomMatrix(this.hiddenSize, this.hiddenSize),
            bi: this._randomVector(this.hiddenSize),

            // Forget gate
            Wif: this._randomMatrix(this.hiddenSize, this.inputSize),
            Whf: this._randomMatrix(this.hiddenSize, this.hiddenSize),
            bf: this._randomVector(this.hiddenSize),

            // Cell gate
            Wig: this._randomMatrix(this.hiddenSize, this.inputSize),
            Whg: this._randomMatrix(this.hiddenSize, this.hiddenSize),
            bg: this._randomVector(this.hiddenSize),

            // Output gate
            Wio: this._randomMatrix(this.hiddenSize, this.inputSize),
            Who: this._randomMatrix(this.hiddenSize, this.hiddenSize),
            bo: this._randomVector(this.hiddenSize),
        };

        return weights;
    }

    /**
     * Create random matrix with Xavier initialization
     * @private
     */
    _randomMatrix(rows, cols) {
        const scale = Math.sqrt(2.0 / (rows + cols));
        return Array(rows).fill(0).map(() =>
            Array(cols).fill(0).map(() => (Math.random() - 0.5) * 2 * scale),
        );
    }

    /**
     * Create random vector
     * @private
     */
    _randomVector(size) {
        return Array(size).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    }

    /**
     * Sigmoid activation function
     * @private
     */
    _sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Tanh activation function
     * @private
     */
    _tanh(x) {
        return Math.tanh(x);
    }

    /**
     * Matrix-vector multiplication
     * @private
     */
    _matVecMul(matrix, vector) {
        return matrix.map(row =>
            row.reduce((sum, val, i) => sum + val * vector[i], 0),
        );
    }

    /**
     * Vector addition
     * @private
     */
    _vecAdd(a, b) {
        return a.map((val, i) => val + b[i]);
    }

    /**
     * Element-wise vector multiplication
     * @private
     */
    _vecMul(a, b) {
        return a.map((val, i) => val * b[i]);
    }

    /**
     * Forward pass through LSTM cell
     * @param {number[]} input - Input vector
     * @param {number[]} h - Previous hidden state
     * @param {number[]} c - Previous cell state
     * @returns {{h: number[], c: number[]}} New hidden and cell states
     */
    forward(input, h, c) {
        const { Wii, Whi, bi, Wif, Whf, bf, Wig, Whg, bg, Wio, Who, bo } = this.weights;

        // Input gate
        let i_t = this._vecAdd(this._matVecMul(Wii, input), this._matVecMul(Whi, h));
        i_t = this._vecAdd(i_t, bi).map(x => this._sigmoid(x));

        // Forget gate
        let f_t = this._vecAdd(this._matVecMul(Wif, input), this._matVecMul(Whf, h));
        f_t = this._vecAdd(f_t, bf).map(x => this._sigmoid(x));

        // Cell gate
        let g_t = this._vecAdd(this._matVecMul(Wig, input), this._matVecMul(Whg, h));
        g_t = this._vecAdd(g_t, bg).map(x => this._tanh(x));

        // Output gate
        let o_t = this._vecAdd(this._matVecMul(Wio, input), this._matVecMul(Who, h));
        o_t = this._vecAdd(o_t, bo).map(x => this._sigmoid(x));

        // New cell state
        const c_new = this._vecAdd(this._vecMul(f_t, c), this._vecMul(i_t, g_t));

        // New hidden state
        const h_new = this._vecMul(o_t, c_new.map(x => this._tanh(x)));

        return { h: h_new, c: c_new };
    }
}

/**
 * Multi-layer LSTM network
 */
export class LSTMNetwork {
    /**
     * @param {string} modelType - Type of model (from config)
     */
    constructor(modelType) {
        this.config = getNeuralModelConfig(modelType);
        this.modelType = modelType;

        // Create LSTM layers
        this.layers = [];
        for (let i = 0; i < this.config.numLayers; i++) {
            const inputSize = i === 0 ? this.config.inputSize : this.config.hiddenSize;
            this.layers.push(new LSTMCell(inputSize, this.config.hiddenSize));
        }

        // Output projection layer
        this.outputWeights = this._initializeOutputWeights();

        // Initialize states
        this.resetStates();
    }

    /**
     * Initialize output projection weights
     * @private
     */
    _initializeOutputWeights() {
        const scale = Math.sqrt(2.0 / (this.config.hiddenSize + this.config.outputSize));
        return Array(this.config.outputSize).fill(0).map(() =>
            Array(this.config.hiddenSize).fill(0).map(() => (Math.random() - 0.5) * 2 * scale),
        );
    }

    /**
     * Reset hidden and cell states to zero
     */
    resetStates() {
        this.hiddenStates = this.layers.map(() =>
            Array(this.config.hiddenSize).fill(0),
        );
        this.cellStates = this.layers.map(() =>
            Array(this.config.hiddenSize).fill(0),
        );
    }

    /**
     * Forward pass through the network
     * @param {number[]} input - Input vector
     * @returns {number[]} Output vector
     */
    forward(input) {
        let layerInput = input;

        // Pass through each LSTM layer
        for (let i = 0; i < this.layers.length; i++) {
            const { h, c } = this.layers[i].forward(
                layerInput,
                this.hiddenStates[i],
                this.cellStates[i],
            );

            this.hiddenStates[i] = h;
            this.cellStates[i] = c;
            layerInput = h;
        }

        // Project to output size
        const output = this.outputWeights.map(row =>
            row.reduce((sum, val, i) => sum + val * this.hiddenStates[this.layers.length - 1][i], 0),
        );

        return output;
    }

    /**
     * Process a sequence of inputs
     * @param {number[][]} sequence - Array of input vectors
     * @returns {number[][]} Array of output vectors
     */
    forwardSequence(sequence) {
        this.resetStates();
        return sequence.map(input => this.forward(input));
    }

    /**
     * Get current state for persistence
     * @returns {Object} State object
     */
    getState() {
        return {
            modelType: this.modelType,
            hiddenStates: this.hiddenStates.map(s => [...s]),
            cellStates: this.cellStates.map(s => [...s]),
            timestamp: Date.now(),
        };
    }

    /**
     * Restore state from persistence
     * @param {Object} state - State object
     */
    setState(state) {
        if (state.modelType !== this.modelType) {
            console.warn(`State model type mismatch: ${state.modelType} vs ${this.modelType}`);
        }
        this.hiddenStates = state.hiddenStates.map(s => [...s]);
        this.cellStates = state.cellStates.map(s => [...s]);
    }
}

/**
 * Utility functions for working with neural networks
 */

/**
 * Normalize a vector to have zero mean and unit variance
 * @param {number[]} vector - Input vector
 * @returns {number[]} Normalized vector
 */
export function normalizeVector(vector) {
    const mean = vector.reduce((sum, val) => sum + val, 0) / vector.length;
    const variance = vector.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / vector.length;
    const std = Math.sqrt(variance + 1e-8); // Add epsilon for numerical stability

    return vector.map(val => (val - mean) / std);
}

/**
 * Calculate cosine similarity between two vectors
 * @param {number[]} a - First vector
 * @param {number[]} b - Second vector
 * @returns {number} Cosine similarity [-1, 1]
 */
export function cosineSimilarity(a, b) {
    if (a.length !== b.length) {
        throw new Error('Vectors must have the same length');
    }

    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

    return dotProduct / (magnitudeA * magnitudeB + 1e-8);
}

/**
 * Pad or truncate sequence to fixed length
 * @param {number[][]} sequence - Input sequence
 * @param {number} targetLength - Desired length
 * @param {number} vectorSize - Size of each vector
 * @returns {number[][]} Padded/truncated sequence
 */
export function padSequence(sequence, targetLength, vectorSize) {
    if (sequence.length === targetLength) {
        return sequence;
    }

    if (sequence.length > targetLength) {
        // Take the most recent items
        return sequence.slice(-targetLength);
    }

    // Pad with zero vectors at the beginning
    const padding = Array(targetLength - sequence.length).fill(0).map(() =>
        Array(vectorSize).fill(0),
    );
    return [...padding, ...sequence];
}

export default {
    LSTMNetwork,
    normalizeVector,
    cosineSimilarity,
    padSequence,
};
