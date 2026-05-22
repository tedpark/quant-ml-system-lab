from quant_ml_lab.experiments import JsonlExperimentTracker, LocalModelRegistry, ModelRegistryEntry


def test_jsonl_experiment_tracker_round_trip(tmp_path):
    tracker = JsonlExperimentTracker(tmp_path / "runs.jsonl")

    run = tracker.log_run(
        name="demo",
        params={"lookback": 20},
        metrics={"sharpe": 0.1},
        tags={"dataset": "synthetic"},
    )
    loaded = tracker.load_runs()

    assert len(loaded) == 1
    assert loaded[0].run_id == run.run_id
    assert loaded[0].metrics["sharpe"] == 0.1


def test_local_model_registry_round_trip(tmp_path):
    registry = LocalModelRegistry(tmp_path / "registry.json")
    entry = ModelRegistryEntry(
        model_name="demo",
        model_version="v1",
        stage="demo",
        metrics={"sharpe": 0.1},
    )

    registry.promote(entry)
    current = registry.current()

    assert current is not None
    assert current.model_name == "demo"
    assert current.disclosure.startswith("Demo registry")
