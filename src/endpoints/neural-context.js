/**
 * Neural context endpoint for RNN/LSTM-based adaptive world building
 * and character evolution tracking.
 */

import express from 'express';
import { CharacterEvolutionTracker } from '../neural/character-evolution.js';
import { WorldStateEvolutionTracker } from '../neural/world-evolution.js';
import { NEURAL_MODEL_CONFIGS } from '../neural/rnn-config.js';

export const router = express.Router();

// Global instances (in production, these would be per-user or cached)
const characterTracker = new CharacterEvolutionTracker();
const worldTracker = new WorldStateEvolutionTracker();

/**
 * Get available neural models and their configurations
 */
router.get('/models', (request, response) => {
    try {
        const models = Object.entries(NEURAL_MODEL_CONFIGS).map(([key, config]) => ({
            id: key,
            description: config.description,
            enabled: config.enabled,
            parameters: {
                hiddenSize: config.hiddenSize,
                numLayers: config.numLayers,
                sequenceLength: config.sequenceLength,
            },
        }));

        return response.json({
            models,
            timestamp: Date.now(),
        });
    } catch (error) {
        console.error('Error getting neural models:', error);
        return response.status(500).json({ error: error.message });
    }
});

/**
 * Process a character interaction for evolution tracking
 */
router.post('/character/process', async (request, response) => {
    try {
        const { characterId, message, metadata } = request.body;

        if (!characterId || !message) {
            return response.status(400).json({
                error: 'Missing required fields: characterId and message',
            });
        }

        const result = await characterTracker.processInteraction(
            characterId,
            message,
            metadata || {}
        );

        return response.json(result);
    } catch (error) {
        console.error('Error processing character interaction:', error);
        return response.status(500).json({ error: error.message });
    }
});

/**
 * Get character evolution summary
 */
router.get('/character/:characterId/summary', (request, response) => {
    try {
        const { characterId } = request.params;
        const summary = characterTracker.getEvolutionSummary(characterId);

        if (!summary) {
            return response.status(404).json({
                error: 'No evolution data found for this character',
            });
        }

        return response.json(summary);
    } catch (error) {
        console.error('Error getting character summary:', error);
        return response.status(500).json({ error: error.message });
    }
});

/**
 * Predict character response characteristics
 */
router.post('/character/:characterId/predict', async (request, response) => {
    try {
        const { characterId } = request.params;
        const { context } = request.body;

        if (!context) {
            return response.status(400).json({
                error: 'Missing required field: context',
            });
        }

        const prediction = await characterTracker.predictResponse(characterId, context);
        return response.json(prediction);
    } catch (error) {
        console.error('Error predicting character response:', error);
        return response.status(500).json({ error: error.message });
    }
});

/**
 * Reset character evolution data
 */
router.delete('/character/:characterId', (request, response) => {
    try {
        const { characterId } = request.params;
        characterTracker.resetCharacter(characterId);

        return response.json({
            success: true,
            message: `Reset evolution data for character ${characterId}`,
        });
    } catch (error) {
        console.error('Error resetting character:', error);
        return response.status(500).json({ error: error.message });
    }
});

/**
 * Export character state
 */
router.get('/character/:characterId/export', (request, response) => {
    try {
        const { characterId } = request.params;
        const exportedState = characterTracker.exportCharacterState(characterId);

        if (!exportedState) {
            return response.status(404).json({
                error: 'No state found for this character',
            });
        }

        return response.json(exportedState);
    } catch (error) {
        console.error('Error exporting character state:', error);
        return response.status(500).json({ error: error.message });
    }
});

/**
 * Import character state
 */
router.post('/character/import', (request, response) => {
    try {
        const { state } = request.body;

        if (!state || !state.characterId) {
            return response.status(400).json({
                error: 'Invalid state data',
            });
        }

        characterTracker.importCharacterState(state);

        return response.json({
            success: true,
            characterId: state.characterId,
            message: 'Character state imported successfully',
        });
    } catch (error) {
        console.error('Error importing character state:', error);
        return response.status(500).json({ error: error.message });
    }
});

/**
 * Process a world event for state evolution
 */
router.post('/world/process', async (request, response) => {
    try {
        const { worldId, event, metadata } = request.body;

        if (!worldId || !event) {
            return response.status(400).json({
                error: 'Missing required fields: worldId and event',
            });
        }

        const result = await worldTracker.processWorldEvent(
            worldId,
            event,
            metadata || {}
        );

        return response.json(result);
    } catch (error) {
        console.error('Error processing world event:', error);
        return response.status(500).json({ error: error.message });
    }
});

/**
 * Get world state summary
 */
router.get('/world/:worldId/summary', (request, response) => {
    try {
        const { worldId } = request.params;
        const summary = worldTracker.getWorldSummary(worldId);

        if (!summary) {
            return response.status(404).json({
                error: 'No world state data found',
            });
        }

        return response.json(summary);
    } catch (error) {
        console.error('Error getting world summary:', error);
        return response.status(500).json({ error: error.message });
    }
});

/**
 * Get entity relationships for a world
 */
router.get('/world/:worldId/entities', (request, response) => {
    try {
        const { worldId } = request.params;
        const entities = worldTracker.getEntityRelationships(worldId);

        return response.json({
            worldId,
            entities,
            count: entities.length,
        });
    } catch (error) {
        console.error('Error getting entity relationships:', error);
        return response.status(500).json({ error: error.message });
    }
});

/**
 * Predict next world state
 */
router.post('/world/:worldId/predict', async (request, response) => {
    try {
        const { worldId } = request.params;
        const { context } = request.body;

        if (!context) {
            return response.status(400).json({
                error: 'Missing required field: context',
            });
        }

        const prediction = await worldTracker.predictNextState(worldId, context);
        return response.json(prediction);
    } catch (error) {
        console.error('Error predicting world state:', error);
        return response.status(500).json({ error: error.message });
    }
});

/**
 * Export world state
 */
router.get('/world/:worldId/export', (request, response) => {
    try {
        const { worldId } = request.params;
        const exportedState = worldTracker.exportWorldState(worldId);

        if (!exportedState) {
            return response.status(404).json({
                error: 'No state found for this world',
            });
        }

        return response.json(exportedState);
    } catch (error) {
        console.error('Error exporting world state:', error);
        return response.status(500).json({ error: error.message });
    }
});

/**
 * Import world state
 */
router.post('/world/import', (request, response) => {
    try {
        const { state } = request.body;

        if (!state || !state.worldId) {
            return response.status(400).json({
                error: 'Invalid state data',
            });
        }

        worldTracker.importWorldState(state);

        return response.json({
            success: true,
            worldId: state.worldId,
            message: 'World state imported successfully',
        });
    } catch (error) {
        console.error('Error importing world state:', error);
        return response.status(500).json({ error: error.message });
    }
});

/**
 * Reset world state
 */
router.delete('/world/:worldId', (request, response) => {
    try {
        const { worldId } = request.params;
        worldTracker.resetWorld(worldId);

        return response.json({
            success: true,
            message: `Reset state for world ${worldId}`,
        });
    } catch (error) {
        console.error('Error resetting world:', error);
        return response.status(500).json({ error: error.message });
    }
});

/**
 * Health check endpoint
 */
router.get('/health', (request, response) => {
    return response.json({
        status: 'ok',
        features: {
            characterEvolution: true,
            worldEvolution: true,
        },
        timestamp: Date.now(),
    });
});
