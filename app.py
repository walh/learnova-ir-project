
import streamlit as st
import pandas as pd
import re
import nltk
import tempfile
import html
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from urllib.parse import urlparse
import pyterrier as pt

st.set_page_config(
    page_title="Learnova | Find your next skill",
    page_icon="🎓",
    layout="wide"
)

# ==============================
# WEBSITE DESIGN
# ==============================

st.markdown("""
<style>

.stApp {
    background: #f5f8ff;
    color: #142344;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        115deg, #09265c, #0056d2 75%, #397df4
    );
    padding: 44px;
    border-radius: 24px;
    color: white;
    margin-bottom: 20px;
}

.hero:after {
    content: '';
    position: absolute;
    width: 280px;
    height: 280px;
    border: 40px solid #ffffff15;
    border-radius: 50%;
    right: -70px;
    top: -90px;
    pointer-events: none;
}

.hero h1 {
    color: white;
    font-size: clamp(32px, 5vw, 56px);
    letter-spacing: -2px;
    line-height: 1.08;
    margin: 8px 0;
    max-width: 800px;
}

.hero p {
    color: #dceaff;
    font-size: 18px;
}

.eyebrow {
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
    color: #b8d6ff;
}

.course {
    background: white;
    border: 1px solid #dce5f5;
    border-radius: 16px;
    padding: 20px;
    min-height: 190px;
}

.course h3 {
    color: #142344;
    margin: 8px 0;
}

.meta {
    color: #526481;
    font-size: 14px;
}

.tag {
    display: inline-block;
    background: #e9f1ff;
    color: #164c9e;
    padding: 4px 9px;
    border-radius: 8px;
    font-size: 12px;
    margin: 3px;
}

.stButton button {
    border-radius: 10px;
}

[data-testid="stMetric"] {
    background: white;
    border-radius: 14px;
    padding: 16px;
    border: 1px solid #dce5f5;
}

.st-key-topic0 button,
.st-key-topic3 button,
.st-key-topic6 button {
    background: #e8f0ff !important;
    color: #17469b !important;
}

.st-key-topic1 button,
.st-key-topic5 button,
.st-key-topic8 button {
    background: #f0e8ff !important;
    color: #6940a0 !important;
}

.st-key-topic2 button,
.st-key-topic7 button {
    background: #e2f4eb !important;
    color: #16664c !important;
}

.st-key-topic4 button {
    background: #ffeadf !important;
    color: #a44c27 !important;
}

[class*="st-key-topic"] button {
    min-height: 76px;
    border: 1px solid #d7e2f2;
    font-weight: 650;
    border-radius: 16px;
}

.st-key-searchbox input {
    font-size: 17px;
    min-height: 48px;
}

.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: #0056d2;
}

</style>
""", unsafe_allow_html=True)


# ==============================
# LOAD DATA AND SEARCH ENGINE
# ==============================

@st.cache_resource(
    show_spinner="Preparing your learning experience..."
)
def setup():

    data = pd.read_csv(
        "cleaned_dataset_restored.csv",
        dtype=str,
        keep_default_na=False
    )

    coll = pd.read_csv(
        "collection_restored.csv",
        dtype=str,
        keep_default_na=False
    )

    normalize = lambda v: " ".join(v.split()).casefold()

    if (
        "Course Name" not in data
        or "title" not in coll
        or len(data) != len(coll)
        or data["Course Name"].map(normalize).tolist()
        != coll["title"].map(normalize).tolist()
    ):
        st.error(
            f"Dataset alignment failed: "
            f"metadata rows={len(data)}, "
            f"collection rows={len(coll)}"
        )
        st.stop()

    data["docno"] = coll.docno

    data["rating_num"] = pd.to_numeric(
        data["Course Rating"],
        errors="coerce"
    )

    try:
        sw = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        sw = set(stopwords.words("english"))

    stem = PorterStemmer()

    def prep(s):
        s = re.sub(
            r"https?://\S+|www\.\S+",
            "",
            str(s).lower()
        )

        s = re.sub(r"[^\w\s]", " ", s)

        return " ".join(
            stem.stem(w)
            for w in s.split()
            if w not in sw
        )

    docs = [
        {
            "docno": r.docno,
            "text": prep(r.title + " " + r.text)
        }
        for r in coll.itertuples()
    ]

    if not pt.java.started():
        pt.java.init()

    idx = pt.terrier.IterDictIndexer(
        tempfile.mkdtemp(),
        stemmer=None,
        stopwords=None
    ).index(docs)

    return data.set_index("docno"), idx, prep


data, idx, prep = setup()


# ==============================
# SESSION STATE
# ==============================

if "saved" not in st.session_state:
    st.session_state.saved = []

if "query" not in st.session_state:
    st.session_state.query = ""


# ==============================
# SEARCH FUNCTION
# ==============================

@st.cache_data(show_spinner=False)
def search(q, model):

    processed = prep(q)

    if not processed:
        return pd.DataFrame(
            columns=["docno", "score", "rank"]
        )

    retriever = pt.terrier.Retriever(
        idx,
        wmodel=model,
        num_results=100,
        controls={"applypipeline": "off"}
    )

    return retriever.transform(
        pd.DataFrame([
            {
                "qid": "1",
                "query": processed
            }
        ])
    )


def esc(x):
    return html.escape(str(x))


# ==============================
# COURSE CARDS
# ==============================

def course(doc, rank=None, score=None, key=""):

    r = data.loc[doc]

    title = str(r["Course Name"])
    rating = r["Course Rating"]
    level = str(r["Difficulty Level"]).title()
    url = str(r["Course URL"])
    skills = str(r["Skills"])
    university = str(r["University"]).title()

    rank_text = f"#{rank} · " if rank else ""

    st.markdown(
        f"""
        <div class="course">
            <div class="meta">
                {rank_text}{esc(university)}
            </div>

            <h3>{esc(title)}</h3>

            <span class="tag">
                ★ {esc(rating)}
            </span>

            <span class="tag">
                {esc(level)}
            </span>

            <p class="meta">
                {esc(skills[:160])}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if score is not None:
        st.caption(
            f"Retrieval score: {score:.4f} · {doc}"
        )

    a, b = st.columns([3, 1])

    parsed = urlparse(url)

    if (
        parsed.scheme == "https"
        and parsed.hostname in (
            "www.coursera.org",
            "coursera.org"
        )
    ):
        a.link_button(
            "View on Coursera ↗",
            url,
            use_container_width=True
        )

    label = (
        "Saved ✓"
        if doc in st.session_state.saved
        else "Save ♡"
    )

    if b.button(label, key=f"{key}-{doc}"):

        if doc in st.session_state.saved:
            st.session_state.saved.remove(doc)
        else:
            st.session_state.saved.append(doc)

        st.rerun()

    with st.expander("Course details"):
        st.write(r["Course Description"])


# ==============================
# NAVIGATION
# ==============================

header, link = st.columns([5, 1])

header.markdown("### ◈ Learnova")

link.link_button(
    "Visit Coursera ↗",
    "https://www.coursera.org/"
)


# ==============================
# HERO SECTION
# ==============================

st.markdown(
    f"""
    <div class="hero">

        <div class="eyebrow">
            LEARN WITHOUT LIMITS
        </div>

        <h1>
            Big ambitions start<br>
            with small steps.
        </h1>

        <p>
            Discover your next passion.
            Build new skills at your own pace.
        </p>

        <p style="font-size:14px">
            {len(data):,} courses
            &nbsp; · &nbsp;
            {data.University.nunique()} learning providers
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ==============================
# MAIN TABS
# ==============================

discover, saved = st.tabs([
    "Discover courses",
    "My learning"
])


# ==============================
# DISCOVER COURSES
# ==============================

with discover:

    st.markdown(
        "### What would you like to learn today?"
    )

    q = st.text_input(
        "Search courses",
        key="query",
        placeholder=(
            "Try: I want to learn how to build websites"
        ),
        label_visibility="collapsed"
    )

    st.caption(
        "Search by a skill, a subject, "
        "or your next career goal."
    )


    # ==============================
    # EXPLORE BY TOPIC
    # ==============================

    st.markdown("### Explore by topic")

    topics = [
        (
            "📊 Data Science",
            "data science"
        ),
        (
            "✨ Artificial Intelligence",
            "artificial intelligence machine learning"
        ),
        (
            "💼 Business",
            "business management"
        ),
        (
            "💻 Computer Science",
            "computer science programming"
        ),
        (
            "🛡️ Cybersecurity",
            "cybersecurity"
        ),
        (
            "🎨 Arts & Design",
            "art design"
        ),
        (
            "🌱 Personal Development",
            "personal development communication"
        ),
        (
            "❤️ Health & Wellness",
            "health wellness"
        ),
        (
            "🌍 Languages",
            "language learning"
        )
    ]

    for start in range(0, len(topics), 3):

        for i, col in enumerate(
            st.columns(3),
            start
        ):

            label, term = topics[i]

            with col:

                with st.container(key=f"topic{i}"):

                    st.button(
                        label,
                        key=f"pick{i}",
                        use_container_width=True,
                        on_click=lambda v=term:
                            st.session_state.update(query=v)
                    )


    # ==============================
    # LEARNING PATHWAYS
    # ==============================

    st.markdown(
        "### Find your next opportunity"
    )

    a, b, c = st.columns(3)

    pathways = [
        (
            a,
            "🚀 Start a new career",
            "career development"
        ),
        (
            b,
            "🧑‍💻 Learn to code",
            "python programming"
        ),
        (
            c,
            "🎤 Build confidence",
            "public speaking communication"
        )
    ]

    for col, label, term in pathways:

        col.button(
            label,
            use_container_width=True,
            on_click=lambda v=term:
                st.session_state.update(query=v)
        )


    # ==============================
    # COURSE FILTERS
    # ==============================

    with st.expander("Refine your search"):

        a, b, c = st.columns(3)

        levels = a.multiselect(
            "Difficulty",
            sorted(
                x for x in
                data["Difficulty Level"].unique()
                if x
            )
        )

        providers = b.multiselect(
            "Learning provider",
            sorted(
                x for x in data.University.unique()
                if x
            )
        )

        minimum = c.selectbox(
            "Minimum rating",
            [0.0, 4.0, 4.5, 4.8],
            format_func=lambda x:
                "Any rating"
                if not x
                else f"{x:.1f} stars"
        )

        model = st.radio(
            "Advanced search: ranking method",
            ["BM25", "TF_IDF"],
            format_func=lambda x:
                x.replace("_", "-"),
            horizontal=True
        )


    # ==============================
    # SEARCH RESULTS
    # ==============================

    term = q.strip() or "python programming"

    st.markdown(
        "### "
        + (
            "Search results"
            if q.strip()
            else "Start exploring"
        )
    )

    res = search(term, model)

    selected = []

    for r in res.itertuples():

        item = data.loc[r.docno]

        if (
            levels
            and item["Difficulty Level"] not in levels
        ):
            continue

        if (
            providers
            and item.University not in providers
        ):
            continue

        if minimum and not item.rating_num >= minimum:
            continue

        selected.append(r)

    st.caption(
        f"{len(selected)} matching courses "
        "in the top 100 search results"
    )

    if not selected:
        st.info(
            "No matching courses. "
            "Try a different search or relax your filters."
        )

    limit = st.selectbox(
        "Courses to display",
        [6, 12, 24, 48],
        index=0
    )

    cols = st.columns(2)

    for i, r in enumerate(selected[:limit]):

        with cols[i % 2]:

            course(
                r.docno,
                key="discover"
            )


# ==============================
# MY LEARNING
# ==============================

with saved:

    st.markdown("### Your learning list")

    st.caption(
        "Keep track of courses you want "
        "to explore during this session."
    )

    if not st.session_state.saved:

        st.info(
            "Save a course to start your learning list."
        )

    for d in list(st.session_state.saved):

        course(
            d,
            key="saved"
        )


# ==============================
# COURSERA SECTION
# ==============================

st.divider()

st.markdown(
    "### Continue your learning journey"
)

st.write(
    "Discover courses on Learnova "
    "and continue learning on Coursera."
)

st.link_button(
    "Explore Coursera ↗",
    "https://www.coursera.org/"
)


# ==============================
# FOOTER
# ==============================

st.caption(
    "Learnova is an independent academic project "
    "using Coursera course metadata. "
    "Not affiliated with Coursera. "
    "Course availability and ratings may change."
)
