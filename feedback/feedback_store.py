from uuid import uuid4

from pydantic import BaseModel


class FeedbackSummary(BaseModel):
    response_id: str
    avg_rating: float
    total_count: int


class FeedbackStore:
    def __init__(self, conn):
        self.conn = conn

    def store_feedback(self, response_id: str, rating: int, comment: str = None) -> str:
        if rating < 1 or rating > 5:
            raise ValueError("rating must be between 1 and 5")
        feedback_id = f"fb_{uuid4().hex[:8]}"
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO intellisupport.feedback
                    (feedback_id, response_id, rating, comment)
                VALUES (%s, %s, %s, %s)
                """,
                (feedback_id, response_id, rating, comment),
            )
        self.conn.commit()
        return feedback_id

    def get_feedback_summary(self, response_id: str) -> FeedbackSummary:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(AVG(rating), 0), COUNT(*)
                FROM intellisupport.feedback
                WHERE response_id = %s
                """,
                (response_id,),
            )
            row = cur.fetchone()
        return FeedbackSummary(
            response_id=response_id,
            avg_rating=float(row[0]),
            total_count=int(row[1]),
        )

    def get_low_rated_responses(
        self,
        threshold: float = 2.5,
        limit: int = 10,
    ) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.response_id, r.query_id, AVG(f.rating) AS avg_rating, COUNT(*) AS feedback_count
                FROM intellisupport.responses r
                JOIN intellisupport.feedback f ON f.response_id = r.response_id
                GROUP BY r.response_id, r.query_id
                HAVING AVG(f.rating) < %s
                ORDER BY avg_rating ASC
                LIMIT %s
                """,
                (threshold, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "response_id": row[0],
                "query_id": row[1],
                "avg_rating": float(row[2]),
                "feedback_count": int(row[3]),
            }
            for row in rows
        ]
