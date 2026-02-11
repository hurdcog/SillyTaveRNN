/**
 * Configuration for RNN/LSTM neural network models used in adaptive world building
 * and character development.
 */

/**
 * Model configurations for different neural network tasks
 */
export const NEURAL_MODEL_CONFIGS = {
    'character-evolution': {
        description: 'LSTM model for tracking character personality evolution',
        hiddenSize: 256,
        numLayers: 2,
        inputSize: 384, // Matches common embedding dimensions
        outputSize: 384,
        sequenceLength: 50, // Number of messages to consider
        enabled: true,
    },
    'world-state-evolution': {
        description: 'Sequence-to-sequence model for dynamic world state updates',
        hiddenSize: 512,
        numLayers: 2,
        inputSize: 384,
        outputSize: 384,
        sequenceLength: 100,
        enabled: true,
    },
    'context-relevance': {
        description: 'RNN for predicting context entry relevance given message history',
        hiddenSize: 128,
        numLayers: 1,
        inputSize: 384,
        outputSize: 1, // Relevance score
        sequenceLength: 20,
        enabled: true,
    },
    'narrative-flow': {
        description: 'LSTM for maintaining narrative coherence and story progression',
        hiddenSize: 256,
        numLayers: 2,
        inputSize: 384,
        outputSize: 384,
        sequenceLength: 30,
        enabled: true,
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
 */
export const STATE_CONFIG = {
    maxHistoryLength: 1000, // Maximum number of interactions to keep in history
    stateSaveInterval: 100, // Save state every N interactions
    stateCompressionEnabled: true,
};

export default {
    NEURAL_MODEL_CONFIGS,
    getNeuralModelConfig,
    isNeuralModelEnabled,
    TRAINING_CONFIG,
    STATE_CONFIG,
};
