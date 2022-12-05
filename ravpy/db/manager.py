import sqlalchemy
import sqlalchemy as db
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import create_database as cd
from sqlalchemy_utils import database_exists, get_tables
from sqlalchemy_utils import drop_database as dba

from .models import Base, Subgraph
from ..config import DATABASE_URI


class DBManager:
    def __init__(self):
        self.create_database()
        self.engine = self.connect()
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def get_session(self):
        Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        return Session

    def connect(self):
        engine = db.create_engine(DATABASE_URI, isolation_level='READ UNCOMMITTED')
        Base.metadata.bind = engine
        return engine

    def create_database(self):
        if not database_exists(DATABASE_URI):
            cd(DATABASE_URI)
            print('Database created')

    def drop_database(self):
        if database_exists(DATABASE_URI):
            dba(DATABASE_URI)
            print('Database dropped')

    def create_tables(self):
        """
        Create tables
        """
        Base.metadata.create_all(self.engine, checkfirst=True)

    def add_subgraph(self, **kwargs):
        """
        Create a subgraph and add values
        :param kwargs: subgraph details
        """
        Session = self.get_session()
        with Session.begin() as session:

            subgraph = self.find_subgraph(graph_id=kwargs['graph_id'], subgraph_id=kwargs['subgraph_id'])
            if subgraph is None:
                # create new subgraph
                subgraph = Subgraph()
                for key, value in kwargs.items():
                    setattr(subgraph, key, value)
                session.add(subgraph)
                self.logger.debug("Subgraph created")
            else:
                self.logger.debug("Subgraph available")
            return subgraph

    def find_subgraph(self, graph_id, subgraph_id):
        """
        Find a subgraph
        :param graph_id: Graph id
        :param subgraph_id: subgraph id
        :return: subgraph object
        """
        Session = self.get_session()
        with Session.begin() as session:
            subgraph = (
                session.query(Subgraph).filter(
                    Subgraph.graph_id == graph_id, Subgraph.subgraph_id == subgraph_id,
                ).first()
            )
            return subgraph

    def update_subgraph(self, subgraph, **kwargs):
        """
        Update a subgraph
        :param subgraph: subgraph object
        :param kwargs: details
        :return: updated subgraph object
        """
        Session = self.get_session()
        with Session.begin() as session:
            for key, value in kwargs.items():
                setattr(subgraph, key, value)
            session.add(subgraph)
            return subgraph

    def delete_subgraph(self, obj):
        """
        Delete subgraph object
        :param obj: subgraph object
        :return: None
        """
        Session = self.get_session()
        with Session.begin() as session:
            session.delete(obj)

    def get_subgraphs(self):
        """
        Fetch all subgraphs
        :return: list of subgraphs
        """
        Session = self.get_session()
        with Session.begin() as session:
            return session.query(Subgraph).order_by(Subgraph.created_at.desc()).all()

    def delete_subgraphs(self):
        """
        Delete all subgraphs
        :return: None
        """
        Session = self.get_session()
        with Session.begin() as session:
            session.query(Subgraph).delete()
