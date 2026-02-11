/**
 * Character evolution tracking using LSTM networks.
 * Learns character personality shifts and emotional arcs from chat history.
 */

import { LSTMNetwork, normalizeVector, cosineSimilarity } from './rnn-core.js';
import { getPipeline } from '../transformers.js';

/**
 * Character evolution tracker
 */
export class CharacterEvolutionTracker {
    constructor() {
        this.network = new LSTMNetwork('character-evolution');
        this.characterStates = new Map(); // characterId -> state
        this.interactionHistory = new Map(); // characterId -> history
    }

    /**
     * Convert a message to an embedding vector
     * @param {string} text - Message text
     * @returns {Promise<number[]>} Embedding vector
     * @private
     */
    async _textToEmbedding(text) {
        try {
            const pipeline = await getPipeline('feature-extraction');
            const output = await pipeline(text, { pooling: 'mean', normalize: true });
            return Array.from(output.data);
        } catch (error) {
            console.warn('Failed to get embedding, using random vector:', error);
            // Fallback: return a random vector
            return Array(384).fill(0).map(() => Math.random() - 0.5);
        }
    }

    /**
     * Process a new interaction for a character
     * @param {string} characterId - Character ID
     * @param {string} message - Message text
     * @param {Object} metadata - Additional metadata (emotion, sentiment, etc.)
     * @returns {Promise<Object>} Character state update
     */
    async processInteraction(characterId, message, metadata = {}) {
        // Get or initialize character state
        if (!this.characterStates.has(characterId)) {
            this.characterStates.set(characterId, this.network.getState());
            this.interactionHistory.set(characterId, []);
        }

        // Convert message to embedding
        const embedding = await this._textToEmbedding(message);
        const normalizedEmbedding = normalizeVector(embedding);

        // Get interaction history
        const history = this.interactionHistory.get(characterId);
        history.push({
            embedding: normalizedEmbedding,
            text: message.substring(0, 100), // Store truncated text for debugging
            metadata,
            timestamp: Date.now(),
        });

        // Keep only recent history
        const maxHistory = this.network.config.sequenceLength * 2;
        if (history.length > maxHistory) {
            history.splice(0, history.length - maxHistory);
        }

        // Restore character state
        const state = this.characterStates.get(characterId);
        this.network.setState(state);

        // Forward pass through network
        const output = this.network.forward(normalizedEmbedding);

        // Save updated state
        const newState = this.network.getState();
        this.characterStates.set(characterId, newState);

        // Calculate personality drift (difference from initial state)
        const drift = this._calculatePersonalityDrift(characterId);

        return {
            characterId,
            evolution: output,
            drift,
            interactionCount: history.length,
            timestamp: Date.now(),
        };
    }

    /**
     * Calculate personality drift for a character
     * @param {string} characterId - Character ID
     * @returns {number} Drift magnitude
     * @private
     */
    _calculatePersonalityDrift(characterId) {
        const history = this.interactionHistory.get(characterId);
        if (!history || history.length < 2) {
            return 0;
        }

        // Compare first and latest embeddings
        const first = history[0].embedding;
        const latest = history[history.length - 1].embedding;
        const similarity = cosineSimilarity(first, latest);

        // Drift is inverse of similarity
        return 1 - similarity;
    }

    /**
     * Get character evolution summary
     * @param {string} characterId - Character ID
     * @returns {Object|null} Evolution summary
     */
    getEvolutionSummary(characterId) {
        if (!this.characterStates.has(characterId)) {
            return null;
        }

        const history = this.interactionHistory.get(characterId);
        const state = this.characterStates.get(characterId);
        const drift = this._calculatePersonalityDrift(characterId);

        return {
            characterId,
            interactionCount: history.length,
            drift,
            state,
            recentInteractions: history.slice(-5).map(i => ({
                text: i.text,
                timestamp: i.timestamp,
            })),
        };
    }

    /**
     * Predict next character response characteristics
     * @param {string} characterId - Character ID
     * @param {string} context - Current context/prompt
     * @returns {Promise<Object>} Prediction
     */
    async predictResponse(characterId, context) {
        if (!this.characterStates.has(characterId)) {
            return {
                characterId,
                prediction: null,
                confidence: 0,
                message: 'No evolution data available for this character',
            };
        }

        // Get context embedding
        const contextEmbedding = await this._textToEmbedding(context);
        const normalizedContext = normalizeVector(contextEmbedding);

        // Restore character state
        const state = this.characterStates.get(characterId);
        this.network.setState(state);

        // Get prediction
        const prediction = this.network.forward(normalizedContext);

        // Calculate confidence based on history length
        const history = this.interactionHistory.get(characterId);
        const confidence = Math.min(1, history.length / this.network.config.sequenceLength);

        return {
            characterId,
            prediction,
            confidence,
            interactionCount: history.length,
        };
    }

    /**
     * Reset character evolution data
     * @param {string} characterId - Character ID
     */
    resetCharacter(characterId) {
        this.characterStates.delete(characterId);
        this.interactionHistory.delete(characterId);
    }

    /**
     * Export character state for persistence
     * @param {string} characterId - Character ID
     * @returns {Object|null} Exportable state
     */
    exportCharacterState(characterId) {
        if (!this.characterStates.has(characterId)) {
            return null;
        }

        return {
            characterId,
            state: this.characterStates.get(characterId),
            history: this.interactionHistory.get(characterId),
            exportedAt: Date.now(),
        };
    }

    /**
     * Import character state from persistence
     * @param {Object} exportedState - Exported state
     */
    importCharacterState(exportedState) {
        const { characterId, state, history } = exportedState;
        this.characterStates.set(characterId, state);
        this.interactionHistory.set(characterId, history);
    }
}

export default {
    CharacterEvolutionTracker,
};
