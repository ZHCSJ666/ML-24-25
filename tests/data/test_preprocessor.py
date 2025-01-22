from src.data.commit_chronicle.preprocessor import CommitChroniclePreprocessor


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
