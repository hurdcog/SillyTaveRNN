# Neural Context System for SillyTavern

This module provides RNN/LSTM-based neural network capabilities for adaptive world building, character evolution tracking, and dynamic story progression in SillyTavern.

## Features

### 1. Character Evolution Tracking
Tracks character personality shifts and emotional arcs over time using LSTM networks.

**Capabilities:**
- Real-time learning from character interactions
- Personality drift detection
- Response prediction based on conversation history
- Character state persistence and export/import

**API Endpoints:**
- `POST /api/neural/character/process` - Process a character interaction
- `GET /api/neural/character/:characterId/summary` - Get evolution summary
- `POST /api/neural/character/:characterId/predict` - Predict response characteristics
- `DELETE /api/neural/character/:characterId` - Reset character data
- `GET /api/neural/character/:characterId/export` - Export character state
- `POST /api/neural/character/import` - Import character state

### 2. World State Evolution
Tracks dynamic world states, entity relationships, and story progression using sequence-to-sequence models.

**Capabilities:**
- Event-driven world state updates
- Entity relationship tracking
- World complexity and coherence metrics
- Narrative flow prediction

**API Endpoints:**
- `POST /api/neural/world/process` - Process a world event
- `GET /api/neural/world/:worldId/summary` - Get world summary
- `GET /api/neural/world/:worldId/entities` - Get entity relationships
- `POST /api/neural/world/:worldId/predict` - Predict next world state
- `DELETE /api/neural/world/:worldId` - Reset world data
- `GET /api/neural/world/:worldId/export` - Export world state
- `POST /api/neural/world/import` - Import world state

### 3. Available Neural Models

The system includes four pre-configured neural models:

1. **character-evolution**: 2-layer LSTM (256 hidden units) for character personality tracking
2. **world-state-evolution**: 2-layer LSTM (512 hidden units) for world state dynamics
3. **context-relevance**: 1-layer RNN (128 hidden units) for context entry relevance scoring
4. **narrative-flow**: 2-layer LSTM (256 hidden units) for story coherence

## Usage Examples

### Character Evolution

```javascript
// Process a character interaction
const response = await fetch('/api/neural/character/process', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        characterId: 'seraphina',
        message: 'I am feeling more confident today.',
        metadata: {
            emotion: 'positive',
            sentiment: 0.8
        }
    })
});

const result = await response.json();
console.log('Character evolution:', result.evolution);
console.log('Personality drift:', result.drift);

// Get character summary
const summary = await fetch('/api/neural/character/seraphina/summary');
const characterData = await summary.json();
console.log('Interaction count:', characterData.interactionCount);
console.log('Recent interactions:', characterData.recentInteractions);

// Predict next response characteristics
const prediction = await fetch('/api/neural/character/seraphina/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        context: 'What are your thoughts on the current situation?'
    })
});
```

### World Evolution

```javascript
// Process a world event
const response = await fetch('/api/neural/world/process', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        worldId: 'eldoria',
        event: 'The ancient dragon awakened, shaking the mountains.',
        metadata: {
            entities: ['ancient dragon', 'mountains', 'village'],
            location: 'northern peaks',
            time: 'dawn'
        }
    })
});

const result = await response.json();
console.log('World state:', result.worldState);
console.log('Evolution metrics:', result.evolution);

// Get entity relationships
const entities = await fetch('/api/neural/world/eldoria/entities');
const entityData = await entities.json();
console.log('Top entities:', entityData.entities);

// Get world summary
const summary = await fetch('/api/neural/world/eldoria/summary');
const worldData = await summary.json();
console.log('Complexity:', worldData.evolution.complexity);
console.log('Coherence:', worldData.evolution.coherence);
```

### List Available Models

```javascript
const response = await fetch('/api/neural/models');
const data = await response.json();
console.log('Available models:', data.models);
```

## Integration with Existing Systems

The neural context system integrates with SillyTavern's existing infrastructure:

1. **Vector Embeddings**: Uses the same embedding pipeline (`feature-extraction`) as the vector system
2. **Transformers.js**: Leverages existing ONNX Runtime infrastructure
3. **Character System**: Can be integrated with character definitions and chat history
4. **World Info**: Can enhance world info activation with neural context relevance

## Configuration

Neural model configurations are defined in `src/neural/rnn-config.js`:

```javascript
export const NEURAL_MODEL_CONFIGS = {
    'character-evolution': {
        hiddenSize: 256,
        numLayers: 2,
        inputSize: 384,
        outputSize: 384,
        sequenceLength: 50,
        enabled: true,
    },
    // ... other models
};
```

You can adjust these parameters or disable models by setting `enabled: false`.

## Architecture

The neural system consists of:

- **`src/neural/rnn-config.js`**: Configuration for neural models
- **`src/neural/rnn-core.js`**: Core LSTM/RNN implementation
- **`src/neural/character-evolution.js`**: Character evolution tracker
- **`src/neural/world-evolution.js`**: World state evolution tracker
- **`src/endpoints/neural-context.js`**: REST API endpoints
- **`tests/neural.test.js`**: Unit tests for neural functionality

## Technical Details

### LSTM Architecture

The implementation uses a standard LSTM cell with:
- Input gate
- Forget gate
- Cell gate
- Output gate

State is maintained across forward passes, allowing the network to learn temporal dependencies in character interactions and world events.

### State Persistence

Both character and world states can be exported and imported, allowing:
- Saving learned patterns across sessions
- Transferring character personalities between scenarios
- Backing up world evolution data

### Performance

The JavaScript implementation is lightweight and suitable for real-time interaction processing. For production use with large-scale training, consider:
- Pre-training models offline
- Loading trained weights instead of random initialization
- Using ONNX models for inference

## Future Enhancements

Potential improvements:
1. Pre-trained models for common character archetypes
2. Attention mechanisms for better context selection
3. Multi-modal inputs (text + metadata features)
4. Transfer learning between characters
5. Hierarchical world state models
6. Integration with story generation pipelines

## Testing

Run the neural network tests:

```bash
cd tests
npm run test:unit -- neural.test.js
```

All tests should pass, verifying:
- Vector normalization and similarity
- Sequence padding/truncation
- LSTM network initialization
- Forward passes
- State persistence

## Support

For issues or questions about the neural context system, please refer to:
- The main SillyTavern documentation
- GitHub issues
- Discord community
