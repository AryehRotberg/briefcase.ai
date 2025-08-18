from database.models.policy import PolicyProcessed
from database.postgresql import get_database_session


def get_categories():
    with get_database_session() as session:
        categories = session.query(PolicyProcessed.category).distinct().all()
        return [cat[0] for cat in categories]
