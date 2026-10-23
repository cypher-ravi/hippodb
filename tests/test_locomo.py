from engram.bench import locomo


def test_load_flattens_sessions_in_order(locomo_file):
    [conv] = locomo.load(locomo_file)
    assert conv.speakers == ("Asha", "Ben")
    assert [t.dia_id for t in conv.turns] == ["D1:1", "D1:2", "D2:1"]
    assert conv.turns[2].session == 2
    assert conv.turns[2].session_date == "4:00 pm on 9 June, 2023"


def test_photo_caption_is_kept(locomo_file):
    [conv] = locomo.load(locomo_file)
    assert "[shares a photo of a dog on a beach]" in conv.turns[2].text


def test_answers_are_strings_and_adversarial_has_none(locomo_file):
    [conv] = locomo.load(locomo_file)
    assert conv.questions[1].answer == "2023"
    assert conv.questions[2].category == 5
    assert conv.questions[2].answer == ""
