# Data Contracts

## Questions

JSONL fields: `id`, `family_id`, `data_status`, `question`, `context`, `sparql`.

```json
{"id":"q-001","family_id":"skill-time-01","data_status":"reviewed","question":"Find resources for arterial blood gas interpretation","context":{"level":"Junior","minutes":15},"sparql":"SELECT DISTINCT ?resource WHERE { ... }"}
```

`data_status` is `synthetic` or `reviewed`. The example above specifies a contract; its abbreviated query is not executable. Ranking experiments additionally require `relevance`, a resource-IRI-to-grade map, and may include prior `learner_state` probabilities. References and grades are never serialized into mT5 input. Split by query family; keep paraphrases together.

## Assessments

```json
{"event_id":"a-001","learner_id":"l-001","skill_id":"http://example.org/hlo.owl#K002","correct":true,"timestamp":"2025-01-01T10:00:00+00:00","data_status":"reviewed"}
```

Use scored outcomes and timezone-aware timestamps. Split training and validation by learner. Evaluation is chronological and prequential: predict the next response before updating its mastery state. Duplicate events do not update mastery twice. Parameter changes require replay into a new database.

## Resources

`resources.json` contains `data_status` and a `resources` list. Each record provides `id`, `title`, `skills`, `minutes`, `levels`, `quality`, `difficulty`, `activity`, `location`, `prerequisites`, and `active`. Scores and probabilities are in `[0, 1]`; minutes are positive integers. The RDF catalog represents the same IDs and HLO objective-competency-skill paths. HLO skill codes identify classes; catalog skill instances have the corresponding class as their type.

Levels are `Junior`, `Senior`, and `Fellow`. Duration is `ext:estimatedDurationMinutes`, with domain `hlo:LearningResource`; the source `hlo:durationMinutes` is not redefined.

Catalog RDF and metadata must agree on resource duration, levels and skill paths. Execution uses explicit RDF triples; no OWL reasoner is invoked.

## Policy Transitions

```json
{"episode_id":"e-001","step":0,"state":[0.2,0.8,0.5,0.0],"action":0,"reward":0.1,"next_state":[0.3,0.7,0.4,0.0],"done":false,"allowed_actions":[true,true,false],"next_allowed_actions":[true,true,false],"data_status":"reviewed"}
```

The default four dimensions are mastery, gap priority, normalized available time, and normalized level. Define the three action meanings and reward formula before collecting transitions. Split by episode. Held-out TD error is a training diagnostic, not an estimate of educational policy benefit.
