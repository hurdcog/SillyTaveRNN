/**
 * Configuration for RNN/LSTM neural network models used in adaptive world building
 * and character development.
 */

import { getConfigValue } from '../util.js';

/**
 * Gets a config value with neural-specific path
 * @param {string} key - Config key
 * @param {*} defaultValue - Default value if not found
 * @returns {*} Config value
 */
function getNeuralConfig(key, defaultValue) {
    try {
        return getConfigValue(`extensions.neural.${key}`, defaultValue);
    } catch {
        // Return default value if config is not available (e.g., during tests)
        return defaultValue;
    }
}

/**
 * Model configurations for different neural network tasks
 * Note: Uses getters to delay config loading until needed
 */
export const NEURAL_MODEL_CONFIGS = {
    'character-evolution': {
        description: 'LSTM model for tracking character personality evolution',
        hiddenSize: 256,
        numLayers: 2,
        inputSize: 384, // Matches common embedding dimensions
        outputSize: 384,
        sequenceLength: 50, // Number of messages to consider
        get enabled() {
            return getNeuralConfig('characterEvolution.enabled', true);
        },
    },
    'world-state-evolution': {
        description: 'Sequence-to-sequence model for dynamic world state updates',
        hiddenSize: 512,
        numLayers: 2,
        inputSize: 384,
        outputSize: 384,
        sequenceLength: 100,
        get enabled() {
            return getNeuralConfig('worldEvolution.enabled', true);
        },
    },
    'context-relevance': {
        description: 'RNN for predicting context entry relevance given message history',
        hiddenSize: 128,
        numLayers: 1,
        inputSize: 384,
        outputSize: 1, // Relevance score
        sequenceLength: 20,
        get enabled() {
            return getNeuralConfig('contextRelevance.enabled', true);
        },
    },
    'narrative-flow': {
        description: 'LSTM for maintaining narrative coherence and story progression',
        hiddenSize: 256,
        numLayers: 2,
        inputSize: 384,
        outputSize: 384,
        sequenceLength: 30,
        get enabled() {
            return getNeuralConfig('narrativeFlow.enabled', true);
        },
    },
};

/**
 * Gets the configuration for a specific neural model type
 * @param {string} modelType - Type of the model
 * @returns {Object} Model configuration
 */
export function getNeuralModelConfig(modelType) {
    if (!NEURAL_MODEL_CONFIGS[modelType]) {
        throw new Error(`Unknown neural model type: ${modelType}`);
    }
    return NEURAL_MODEL_CONFIGS[modelType];
}

/**
 * Checks if a neural model type is enabled
 * @param {string} modelType - Type of the model
 * @returns {boolean} Whether the model is enabled
 */
export function isNeuralModelEnabled(modelType) {
    const config = NEURAL_MODEL_CONFIGS[modelType];
    return config && config.enabled;
}

/**
 * Training hyperparameters
 */
export const TRAINING_CONFIG = {
    learningRate: 0.001,
    batchSize: 16,
    epochs: 10,
    validationSplit: 0.2,
    patience: 3, // Early stopping patience
};

/**
 * State management for neural models
 * Note: Uses default values if config is not available
 */
export const STATE_CONFIG = {
    get maxHistoryLength() {
        return getNeuralConfig('characterEvolution.maxHistoryLength', 1000);
    },
    get stateSaveInterval() {
        return getNeuralConfig('characterEvolution.stateSaveInterval', 100);
    },
    stateCompressionEnabled: true,
};

export default {
    NEURAL_MODEL_CONFIGS,
    getNeuralModelConfig,
    isNeuralModelEnabled,
    TRAINING_CONFIG,
    STATE_CONFIG,
};
