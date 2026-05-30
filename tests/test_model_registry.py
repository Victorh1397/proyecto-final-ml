from src.models import get_model_definitions


def test_all_model_registry_contains_15_unique_models() -> None:
    definitions = get_model_definitions(("all",), random_state=42, neural_epochs=1)
    names = [definition.name for definition in definitions]

    expected_new_models = {
        "extra_trees",
        "ada_boost",
        "bagging",
        "knn",
        "svm_rbf",
        "linear_svm",
        "gaussian_nb",
        "ridge_classifier",
        "lightgbm",
        "catboost",
    }

    assert len(names) == 15
    assert len(names) == len(set(names))
    assert expected_new_models.issubset(names)
