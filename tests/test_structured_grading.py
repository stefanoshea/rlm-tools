import pytest

from evals.grade_eval import grade_output


EXPECTED = '{"active":true,"counts":[0,2],"source":"app.py"}'


@pytest.mark.parametrize(("answer", "passed"), [
    ('```json\n{"source":"app.py", "counts":[0.0,2], "active":true}\n```', True),
    (EXPECTED, True),
    ('{"active":1,"counts":[0,2],"source":"app.py"}', False),
    ('{"active":true,"counts":[false,2],"source":"app.py"}', False),
    ('{"active":true,"counts":[2,0],"source":"app.py"}', False),
    ('{"active":true,"counts":[0,2]}', False),
    (EXPECTED[:-1] + ',"legacy":null}', False),
    (EXPECTED[:-1] + ',"source":"app.py"}', False),
    (EXPECTED + ' and possibly legacy.py', False),
    ('{"source":"app.py","counts":[NaN,2],"active":true}', False),
])
def test_json_answer_grades_complete_typed_values(answer, passed):
    grade = grade_output(answer, [{"id": "state", "kind": "json_equals", "values": [EXPECTED]}])
    assert grade["assertions"][0]["passed"] is passed
    assert grade["score"] == int(passed)


def test_invalid_reference_answer_surfaces_spec_error():
    with pytest.raises(ValueError):
        grade_output("{}", [{"id": "state", "kind": "json_equals", "values": ["missing quotes"]}])
