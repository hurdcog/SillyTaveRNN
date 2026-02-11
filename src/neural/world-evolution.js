/**
 * World state evolution using RNN/LSTM networks.
 * Tracks dynamic world state updates and entity relationships.
 */

import { LSTMNetwork, normalizeVector, cosineSimilarity, padSequence } from './rnn-core.js';
import { getPipeline } from '../transformers.js';

/**
 * World state evolution tracker
 */
export class WorldStateEvolutionTracker {
    constructor() {
        this.network = new LSTMNetwork('world-state-evolution');
        this.worldStates = new Map(); // worldId -> state
        this.worldHistory = new Map(); // worldId -> history
        this.entityRelationships = new Map(); // worldId -> Map<entity, relations>
    }

    /**
     * Convert text to embedding vector
     * @param {string} text - Text to embed
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
            return Array(384).fill(0).map(() => Math.random() - 0.5);
        }
    }

    /**
     * Process a world event (story progression, character action, etc.)
     * @param {string} worldId - World/story ID
     * @param {string} event - Event description
     * @param {Object} metadata - Event metadata (entities, location, time, etc.)
     * @returns {Promise<Object>} Updated world state
     */
    async processWorldEvent(worldId, event, metadata = {}) {
        // Initialize world state if needed
        if (!this.worldStates.has(worldId)) {
            this.worldStates.set(worldId, this.network.getState());
            this.worldHistory.set(worldId, []);
            this.entityRelationships.set(worldId, new Map());
        }

        // Convert event to embedding
        const embedding = await this._textToEmbedding(event);
        const normalizedEmbedding = normalizeVector(embedding);

        // Get world history
        const history = this.worldHistory.get(worldId);
        history.push({
            embedding: normalizedEmbedding,
            event: event.substring(0, 200),
            metadata,
            timestamp: Date.now(),
        });

        // Keep only recent history
        const maxHistory = this.network.config.sequenceLength * 2;
        if (history.length > maxHistory) {
            history.splice(0, history.length - maxHistory);
        }

        // Update entity relationships if provided
        if (metadata.entities) {
            this._updateEntityRelationships(worldId, metadata.entities, metadata);
        }

        // Restore world state
        const state = this.worldStates.get(worldId);
        this.network.setState(state);

        // Forward pass through network
        const output = this.network.forward(normalizedEmbedding);

        // Save updated state
        const newState = this.network.getState();
        this.worldStates.set(worldId, newState);

        // Calculate world evolution metrics
        const evolution = this._calculateWorldEvolution(worldId);

        return {
            worldId,
            worldState: output,
            evolution,
            eventCount: history.length,
            timestamp: Date.now(),
        };
    }

    /**
     * Update entity relationships based on event
     * @param {string} worldId - World ID
     * @param {Array<string>} entities - Entities involved
     * @param {Object} metadata - Event metadata
     * @private
     */
    _updateEntityRelationships(worldId, entities, metadata) {
        const relationships = this.entityRelationships.get(worldId);

        for (const entity of entities) {
            if (!relationships.has(entity)) {
                relationships.set(entity, {
                    firstMention: Date.now(),
                    occurrences: 0,
                    relatedEntities: new Map(),
                });
            }

            const entityData = relationships.get(entity);
            entityData.occurrences++;
            entityData.lastMention = Date.now();

            // Track co-occurrences with other entities
            for (const otherEntity of entities) {
                if (otherEntity !== entity) {
                    if (!entityData.relatedEntities.has(otherEntity)) {
                        entityData.relatedEntities.set(otherEntity, 0);
                    }
                    entityData.relatedEntities.set(
                        otherEntity,
                        entityData.relatedEntities.get(otherEntity) + 1
                    );
                }
            }
        }
    }

    /**
     * Calculate world evolution metrics
     * @param {string} worldId - World ID
     * @returns {Object} Evolution metrics
     * @private
     */
    _calculateWorldEvolution(worldId) {
        const history = this.worldHistory.get(worldId);
        if (!history || history.length < 2) {
            return {
                complexity: 0,
                coherence: 1,
                drift: 0,
            };
        }

        // Complexity: number of unique entities and events
        const relationships = this.entityRelationships.get(worldId);
        const complexity = relationships.size / 100; // Normalized

        // Coherence: similarity between consecutive events
        let coherenceSum = 0;
        for (let i = 1; i < Math.min(10, history.length); i++) {
            const similarity = cosineSimilarity(
                history[history.length - i - 1].embedding,
                history[history.length - i].embedding
            );
            coherenceSum += similarity;
        }
        const coherence = coherenceSum / Math.min(9, history.length - 1);

        // Drift: change from initial state
        const first = history[0].embedding;
        const latest = history[history.length - 1].embedding;
        const drift = 1 - cosineSimilarity(first, latest);

        return {
            complexity: Math.min(1, complexity),
            coherence: Math.max(0, coherence),
            drift: Math.max(0, drift),
        };
    }

    /**
     * Predict next likely world state given context
     * @param {string} worldId - World ID
     * @param {string} context - Current context
     * @returns {Promise<Object>} Prediction
     */
    async predictNextState(worldId, context) {
        if (!this.worldStates.has(worldId)) {
            return {
                worldId,
                prediction: null,
                confidence: 0,
                message: 'No world state data available',
            };
        }

        // Get context embedding
        const contextEmbedding = await this._textToEmbedding(context);
        const normalizedContext = normalizeVector(contextEmbedding);

        // Restore world state
        const state = this.worldStates.get(worldId);
        this.network.setState(state);

        // Get prediction
        const prediction = this.network.forward(normalizedContext);

        // Calculate confidence
        const history = this.worldHistory.get(worldId);
        const confidence = Math.min(1, history.length / this.network.config.sequenceLength);

        return {
            worldId,
            prediction,
            confidence,
            eventCount: history.length,
        };
    }

    /**
     * Get entity relationships for a world
     * @param {string} worldId - World ID
     * @returns {Array<Object>} Entity relationship data
     */
    getEntityRelationships(worldId) {
        if (!this.entityRelationships.has(worldId)) {
            return [];
        }

        const relationships = this.entityRelationships.get(worldId);
        const entities = [];

        for (const [entity, data] of relationships) {
            const related = Array.from(data.relatedEntities.entries())
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5)
                .map(([name, count]) => ({ name, count }));

            entities.push({
                name: entity,
                occurrences: data.occurrences,
                firstMention: data.firstMention,
                lastMention: data.lastMention,
                relatedEntities: related,
            });
        }

        return entities.sort((a, b) => b.occurrences - a.occurrences);
    }

    /**
     * Get world evolution summary
     * @param {string} worldId - World ID
     * @returns {Object|null} World summary
     */
    getWorldSummary(worldId) {
        if (!this.worldStates.has(worldId)) {
            return null;
        }

        const history = this.worldHistory.get(worldId);
        const evolution = this._calculateWorldEvolution(worldId);
        const entities = this.getEntityRelationships(worldId);

        return {
            worldId,
            eventCount: history.length,
            evolution,
            entities: entities.slice(0, 10),
            recentEvents: history.slice(-5).map(e => ({
                event: e.event,
                timestamp: e.timestamp,
            })),
        };
    }

    /**
     * Export world state for persistence
     * @param {string} worldId - World ID
     * @returns {Object|null} Exportable state
     */
    exportWorldState(worldId) {
        if (!this.worldStates.has(worldId)) {
            return null;
        }

        return {
            worldId,
            state: this.worldStates.get(worldId),
            history: this.worldHistory.get(worldId),
            relationships: Array.from(this.entityRelationships.get(worldId).entries()),
            exportedAt: Date.now(),
        };
    }

    /**
     * Import world state from persistence
     * @param {Object} exportedState - Exported state
     */
    importWorldState(exportedState) {
        const { worldId, state, history, relationships } = exportedState;
        this.worldStates.set(worldId, state);
        this.worldHistory.set(worldId, history);
        this.entityRelationships.set(worldId, new Map(relationships));
    }

    /**
     * Reset world state
     * @param {string} worldId - World ID
     */
    resetWorld(worldId) {
        this.worldStates.delete(worldId);
        this.worldHistory.delete(worldId);
        this.entityRelationships.delete(worldId);
    }
}

export default {
    WorldStateEvolutionTracker,
};
