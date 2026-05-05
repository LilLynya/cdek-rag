from app.knowledge import KnowledgeBase


def test_loads_all_five_documents():
    kb = KnowledgeBase(data_dir="data")
    assert len(kb.documents) == 5
    sources = {d.metadata["source"] for d in kb.documents}
    assert sources == {
        "general_info.txt",
        "deadlines.txt",
        "benefits.txt",
        "germany_rules.txt",
        "france_rules.txt",
    }


def test_country_metadata_assigned_only_to_country_files():
    kb = KnowledgeBase(data_dir="data")
    by_source = {d.metadata["source"]: d.metadata.get("country") for d in kb.documents}
    assert by_source["germany_rules.txt"] == "germany"
    assert by_source["france_rules.txt"] == "france"
    assert by_source["general_info.txt"] is None
    assert by_source["deadlines.txt"] is None
    assert by_source["benefits.txt"] is None


def test_country_filter_excludes_other_country_documents():
    kb = KnowledgeBase(data_dir="data")
    docs = kb.search("ставка стипендии и налог", country="germany")
    sources = {d.metadata["source"] for d in docs}
    assert "germany_rules.txt" in sources
    assert "france_rules.txt" not in sources


def test_country_filter_for_france():
    kb = KnowledgeBase(data_dir="data")
    docs = kb.search("какой рабочий день", country="france")
    sources = {d.metadata["source"] for d in docs}
    assert "france_rules.txt" in sources
    assert "germany_rules.txt" not in sources


def test_search_without_country_returns_results():
    kb = KnowledgeBase(data_dir="data")
    docs = kb.search("когда дедлайн подачи документов")
    sources = [d.metadata["source"] for d in docs]
    assert "deadlines.txt" in sources
