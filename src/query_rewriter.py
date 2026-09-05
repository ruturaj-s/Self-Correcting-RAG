import re


# ============================================================
# MEMORY OF THE LAST STANDALONE QUESTION
# ============================================================

LAST_QUESTION = ""


# ============================================================
# EXTRACT TOPIC FROM A QUESTION
# ============================================================

def extract_topic(question):
    if not question:
        return ""

    question = str(question).strip()
    question = question.rstrip("?").strip()

    prefixes = [
        "what is ",
        "what are ",
        "who is ",
        "who are ",
        "where is ",
        "where are ",
        "when is ",
        "when was ",
        "when were ",
        "how is ",
        "how are ",
        "explain ",
        "define ",
        "tell me about ",
        "describe ",
    ]

    lower_question = question.lower()

    for prefix in prefixes:
        if lower_question.startswith(prefix):
            return question[len(prefix):].strip()

    return question


# ============================================================
# DETECT FOLLOW-UP QUESTION
# ============================================================

def is_follow_up_question(question):

    if not question:
        return False

    q = question.lower().strip()

    patterns = [

        # Pronouns
        r"\bits\b",
        r"\bit\b",
        r"\bthey\b",
        r"\bthem\b",
        r"\btheir\b",
        r"\bthis\b",
        r"\bthat\b",
        r"\bthese\b",
        r"\bthose\b",

        # Common follow-ups
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bwhat are its\b",
        r"\bwhat is its\b",
        r"\bhow does it\b",
        r"\bwhy is it\b",
        r"\bwhere is it\b",
        r"\bwhen was it\b",
        r"\bwhat does it\b",
        r"\bhow can it\b",
        r"\bwhy does it\b",
    ]

    for pattern in patterns:
        if re.search(pattern, q):
            return True

    return False


# ============================================================
# DETERMINISTIC FOLLOW-UP REWRITER
# ============================================================

def deterministic_rewrite(question, topic):

    q = question.strip()
    lower_q = q.lower()

    # --------------------------------------------------------
    # MAIN CHARACTERISTICS
    # --------------------------------------------------------

    if re.search(
        r"\bwhat are its main characteristics\b",
        lower_q
    ):
        return f"What are the main characteristics of {topic}?"

    # --------------------------------------------------------
    # CHARACTERISTICS
    # --------------------------------------------------------

    if re.search(
        r"\bwhat are its characteristics\b",
        lower_q
    ):
        return f"What are the characteristics of {topic}?"

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    if re.search(
        r"\bwhat are its features\b",
        lower_q
    ):
        return f"What are the features of {topic}?"

    # --------------------------------------------------------
    # ADVANTAGES
    # --------------------------------------------------------

    if re.search(
        r"\bwhat are its advantages\b",
        lower_q
    ):
        return f"What are the advantages of {topic}?"

    # --------------------------------------------------------
    # DISADVANTAGES
    # --------------------------------------------------------

    if re.search(
        r"\bwhat are its disadvantages\b",
        lower_q
    ):
        return f"What are the disadvantages of {topic}?"

    # --------------------------------------------------------
    # APPLICATIONS
    # --------------------------------------------------------

    if re.search(
        r"\bwhat are its applications\b",
        lower_q
    ):
        return f"What are the applications of {topic}?"

    # --------------------------------------------------------
    # USES
    # --------------------------------------------------------

    if re.search(
        r"\bwhat are its uses\b",
        lower_q
    ):
        return f"What are the uses of {topic}?"

    # --------------------------------------------------------
    # PURPOSE
    # --------------------------------------------------------

    if re.search(
        r"\bwhat is its purpose\b",
        lower_q
    ):
        return f"What is the purpose of {topic}?"

    # --------------------------------------------------------
    # IMPORTANCE
    # --------------------------------------------------------

    if re.search(
        r"\bwhy is it important\b",
        lower_q
    ):
        return f"Why is {topic} important?"

    # --------------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------------

    if re.search(
        r"\bhow does it work\b",
        lower_q
    ):
        return f"How does {topic} work?"

    # --------------------------------------------------------
    # DEFINITION
    # --------------------------------------------------------

    if re.search(
        r"\bwhat is it\b",
        lower_q
    ):
        return f"What is {topic}?"

    # --------------------------------------------------------
    # GENERAL "WHAT ABOUT"
    # --------------------------------------------------------

    if lower_q.startswith("what about"):
        remaining = q[len("what about"):].strip()

        if remaining:
            return f"{remaining} about {topic}?"

        return f"Tell me more about {topic}."

    # --------------------------------------------------------
    # GENERAL "HOW ABOUT"
    # --------------------------------------------------------

    if lower_q.startswith("how about"):
        remaining = q[len("how about"):].strip()

        if remaining:
            return f"{remaining} about {topic}?"

        return f"Tell me more about {topic}."

    # --------------------------------------------------------
    # GENERIC PRONOUN REPLACEMENT
    # --------------------------------------------------------

    replacements = [
        (r"\bits\b", f"{topic}'s"),
        (r"\bit\b", topic),
        (r"\bthey\b", topic),
        (r"\bthem\b", topic),
        (r"\btheir\b", f"{topic}'s"),
        (r"\bthis\b", topic),
        (r"\bthat\b", topic),
        (r"\bthese\b", topic),
        (r"\bthose\b", topic),
    ]

    rewritten = q

    for pattern, replacement in replacements:
        rewritten = re.sub(
            pattern,
            replacement,
            rewritten,
            flags=re.IGNORECASE
        )

    return rewritten


# ============================================================
# MAIN FUNCTION
# ============================================================

def rewrite_query(question, chat_history=None):

    global LAST_QUESTION

    if not question:
        return ""

    question = str(question).strip()

    if not question:
        return ""

    # ========================================================
    # FIND PREVIOUS QUESTION FROM CHAT HISTORY
    # ========================================================

    previous_question = ""

    if chat_history:

        # ----------------------------------------------------
        # LIST OF DICTIONARIES
        # ----------------------------------------------------

        if isinstance(chat_history, list):

            for item in reversed(chat_history):

                # Normal:
                # {"role": "user", "content": "..."}
                if isinstance(item, dict):

                    role = item.get("role", "")
                    content = item.get("content", "")

                    if role == "user" and content:

                        previous_question = str(
                            content
                        ).strip()

                        break

                # ------------------------------------------------
                # Tuple style:
                # ("user", "question")
                # ------------------------------------------------

                elif isinstance(item, (list, tuple)):

                    if len(item) >= 2:

                        role = str(item[0]).lower()
                        content = item[1]

                        if (
                            "user" in role
                            and content
                        ):

                            previous_question = str(
                                content
                            ).strip()

                            break

                # ------------------------------------------------
                # Simple string history
                # ------------------------------------------------

                elif isinstance(item, str):

                    if item.strip():

                        previous_question = item.strip()

                        break

    # ========================================================
    # USE INTERNAL MEMORY IF CHAT HISTORY IS EMPTY
    # ========================================================

    if not previous_question and LAST_QUESTION:

        previous_question = LAST_QUESTION

    # ========================================================
    # DETERMINE WHETHER THIS IS A FOLLOW-UP
    # ========================================================

    follow_up = is_follow_up_question(question)

    # ========================================================
    # FOLLOW-UP QUESTION
    # ========================================================

    if follow_up and previous_question:

        topic = extract_topic(previous_question)

        if topic:

            rewritten = deterministic_rewrite(
                question,
                topic
            )

            print()
            print("======================================")
            print("QUERY REWRITING")
            print("======================================")
            print("Original :", question)
            print("Previous :", previous_question)
            print("Topic    :", topic)
            print("Rewritten:", rewritten)
            print("======================================")
            print()

            # Save rewritten question as current context
            LAST_QUESTION = rewritten

            return rewritten

    # ========================================================
    # NORMAL QUESTION
    # ========================================================

    LAST_QUESTION = question

    print()
    print("======================================")
    print("QUERY")
    print("======================================")
    print("Original :", question)
    print("Rewritten:", question)
    print("======================================")
    print()

    return question