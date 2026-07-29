import csv

from src.predict import append_prediction


def test_append_prediction_creates_file_with_header_and_row(tmp_path):
    history_path = tmp_path / "history" / "predictions_history.csv"
    row = {
        "fecha_generacion": "2026-07-28 00:00:00",
        "fecha_objetivo": "2026-07-29",
        "precio_referencia": 250.5,
        "precio_predicho": 260.1,
    }

    append_prediction(history_path, row)

    assert history_path.exists()
    with open(history_path, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    assert reader[0] == list(row.keys())
    assert reader[1] == [str(v) for v in row.values()]
    assert len(reader) == 2


def test_append_prediction_appends_without_repeating_header(tmp_path):
    history_path = tmp_path / "predictions_history.csv"
    row1 = {
        "fecha_generacion": "2026-07-28 00:00:00",
        "fecha_objetivo": "2026-07-29",
        "precio_referencia": 250.5,
        "precio_predicho": 260.1,
    }
    row2 = {
        "fecha_generacion": "2026-07-29 00:00:00",
        "fecha_objetivo": "2026-07-30",
        "precio_referencia": 260.1,
        "precio_predicho": 255.3,
    }

    append_prediction(history_path, row1)
    append_prediction(history_path, row2)

    with open(history_path, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    assert reader[0] == list(row1.keys())
    assert len(reader) == 3
    assert reader[1] == [str(v) for v in row1.values()]
    assert reader[2] == [str(v) for v in row2.values()]
