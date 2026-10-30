from dataclasses import dataclass


@dataclass(frozen=True)
class TurnExpectation:
    required_fragments: tuple[str, ...] = ()
    forbidden_exact_answers: tuple[str, ...] = ()
    question_count: int | None = None
    max_characters: int = 500
    expected_error_fragment: str | None = None

    def __post_init__(self) -> None:
        if self.max_characters <= 0:
            raise ValueError("max_characters должен быть больше нуля.")
        if self.question_count is not None and self.question_count < 0:
            raise ValueError("question_count не может быть отрицательным.")


@dataclass(frozen=True)
class EvalTurn:
    user_text: str
    expectation: TurnExpectation = TurnExpectation()


@dataclass(frozen=True)
class ChatEvalCase:
    case_id: str
    description: str
    turns: tuple[EvalTurn, ...]
    manual_review: str

@dataclass(frozen=True)
class TurnEvalResult:
    case_id: str
    turn_number: int
    user_text: str
    assistant_text: str | None
    error: str | None
    passed: bool
    failures: tuple[str, ...]
    prompt_version: str | None
    history_messages: int | None
    summary_updated: bool | None
    has_summary: bool | None