from src.data.commit_chronicle.preprocessors import CommitChroniclePreprocessor


def test_preprocess_mods():
    """Verifies that different types of file modifications are preprocessed correctly based on
    their `change_type` attribute."""
    preprocessor = CommitChroniclePreprocessor(
        diff_tokenizer=None, msg_tokenizer=None, diff_line_sep="[NL]"
    )  # tokenizers are not relevant

    # check that all mods types work correctly
    modify_mod = {
        "change_type": "MODIFY",
        "old_path": "fname",
        "new_path": "fname",
        "diff": "context 1[NL]context 2[NL]context 3[NL]-old line[NL]+new line[NL]",
    }
    assert preprocessor._preprocess_mods([modify_mod]) == "fname[NL]" + modify_mod["diff"]

    add_mod = {
        "change_type": "ADD",
        "old_path": None,
        "new_path": "fname",
        "diff": "context 1[NL]context 2[NL]context 3[NL]-old line[NL]+new line[NL]",
    }
    assert preprocessor._preprocess_mods([add_mod]) == "new file fname[NL]" + add_mod["diff"]

    delete_mod = {
        "change_type": "DELETE",
        "old_path": "fname",
        "new_path": None,
        "diff": "context 1[NL]context 2[NL]context 3[NL]-old line[NL]+new line[NL]",
    }
    assert (
        preprocessor._preprocess_mods([delete_mod])
        == "deleted file fname[NL]" + delete_mod["diff"]
    )

    rename_mod = {
        "change_type": "RENAME",
        "old_path": "fname1",
        "new_path": "fname2",
        "diff": "context 1[NL]context 2[NL]context 3[NL]-old line[NL]+new line[NL]",
    }
    assert (
        preprocessor._preprocess_mods([rename_mod])
        == "rename from fname1[NL]rename to fname2[NL]" + rename_mod["diff"]
    )

    copy_mod = {
        "change_type": "COPY",
        "old_path": "fname1",
        "new_path": "fname2",
        "diff": "context 1[NL]context 2[NL]context 3[NL]-old line[NL]+new line[NL]",
    }
    assert (
        preprocessor._preprocess_mods([copy_mod])
        == "copy from fname1[NL]copy to fname2[NL]" + copy_mod["diff"]
    )

    # check some mods together
    assert preprocessor._preprocess_mods([modify_mod, modify_mod, add_mod]) == (
        "fname[NL]"
        + modify_mod["diff"]
        + "fname[NL]"
        + modify_mod["diff"]
        + "new file fname[NL]"
        + add_mod["diff"]
    )


def test_get_pos_in_history():
    """Verifies that the calculated positional index of each commit in its historical sequence is
    correct."""
    preprocessor = CommitChroniclePreprocessor(
        diff_tokenizer=None, msg_tokenizer=None
    )  # tokenizers are not relevant
    positions = preprocessor._get_pos_in_history([1, 1, 2, 2, 3])
    assert positions == [0, 1, 0, 1, 0]
    assert preprocessor._num_commits == {1: 2, 2: 2, 3: 1}

    positions = preprocessor._get_pos_in_history([2, 1, 2, 55])
    assert positions == [2, 2, 3, 0]
    assert preprocessor._num_commits == {1: 3, 2: 4, 3: 1, 55: 1}


# def test_process_history(tmp_path):
#     preprocessor = CommitChroniclePreprocessor(
#         diff_tokenizer=None, msg_tokenizer=None
#     )  # tokenizers are not relevant
#
#     with jsonlines.open(f"{tmp_path}/test_file.jsonl", "w") as writer:
#         writer.write_all(
#             [{"author": i, "msg_input_ids": [i]} for i in range(10)]
#             + [{"author": i, "msg_input_ids": [i + 100]} for i in range(5, 15)]
#         )
#
#     preprocessor._process_history(
#         input_path=f"{tmp_path}/test_file.jsonl", output_path=f"{tmp_path}/test_history.json"
#     )
#     with open(f"{tmp_path}/test_history.json") as f:
#         history = json.load(f)
#
#     assert set(history.keys()) == {f"{i}" for i in range(15)}
#     for i in range(5):
#         assert history[f"{i}"] == [[i]]
#     for i in range(5, 10):
#         assert history[f"{i}"] == [[i], [i + 100]]
#     for i in range(10, 15):
#         assert history[f"{i}"] == [[i + 100]]
#

# def test_add_history_to_inputs(tmp_path):
#     preprocessor = CommitChroniclePreprocessor(
#         diff_tokenizer=None, msg_tokenizer=None
#     )  # tokenizers are not relevant
#
#     data = [{"msg_input_ids": f"msg{i}", "author": 0, "pos_in_history": i} for i in range(10)]
#     data += [
#         {"msg_input_ids": f"msg{i + 100}", "author": 1, "pos_in_history": i} for i in range(10)
#     ]
#     with jsonlines.open(f"{tmp_path}/test.jsonl", "w") as writer:
#         writer.write_all(data)
#
#     history = {0: [f"msg{i}" for i in range(10)], 1: [f"msg{i + 100}" for i in range(10)]}
#     with open(f"{tmp_path}/test_history.json", "w") as f:
#         json.dump(history, f)
#
#     preprocessor._add_history_to_inputs(
#         input_path=f"{tmp_path}/test.jsonl",
#         history_path=f"{tmp_path}/test_history.json",
#         output_path=f"{tmp_path}/test_w_history.jsonl",
#         part="test",
#         decoder_context_max_length=100,
#     )
#     with jsonlines.open(f"{tmp_path}/test_w_history.jsonl", "r") as reader:
#         results = [line for line in reader]
#     assert all(
#         [
#             line["history_input_ids"] == [f"msg{i}" for i in range(line["pos_in_history"])]
#             for line in results
#             if line["author"] == 0
#         ]
#     )
#     assert all(
#         [
#             line["history_input_ids"] == [f"msg{i + 100}" for i in range(line["pos_in_history"])]
#             for line in results
#             if line["author"] == 1
#         ]
#     )
#
#     preprocessor._add_history_to_inputs(
#         input_path=f"{tmp_path}/test.jsonl",
#         history_path=f"{tmp_path}/test_history.json",
#         output_path=f"{tmp_path}/test_w_history.jsonl",
#         part="test",
#         decoder_context_max_length=16,
#     )
#     with jsonlines.open(f"{tmp_path}/test_w_history.jsonl", "r") as reader:
#         results = [line for line in reader]
#     assert all(
#         [
#             line["history_input_ids"]
#             == [
#                 f"msg{i}"
#                 for i in range(line["pos_in_history"] - 2, line["pos_in_history"])
#                 if i >= 0
#             ]
#             for line in results
#             if line["author"] == 0
#         ]
#     )
#     assert all(
#         [
#             line["history_input_ids"]
#             == [
#                 f"msg{i + 100}"
#                 for i in range(line["pos_in_history"] - 1, line["pos_in_history"])
#                 if i >= 0
#             ]
#             for line in results
#             if line["author"] == 1
#         ]
#     )
