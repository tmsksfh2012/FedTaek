import syft as sy
from user.interface.IUserEntity import IUserEntity

class ServerEntity:
    def __init__(self, server_name):
        self.server_name  = server_name
        self.server_site  = None
        self.user         = None
        
class ServerHelper:
    def __init__(self, server: ServerEntity):
        self.server = server

    # def get_user_by_id(self, user_id: int) -> DataEntity:
    #     # 여기서는 실제 DB에서 user_id로 유저 정보를 검색하는 로직
    #     # 예제를 위해 가짜 데이터 반환
    #     # 실제로는 SQL 쿼리 실행 혹은 ORM 사용
    #     return DataEntity(user_id=user_id, name="John Doe", email="john@example.com")

    def get_server_name(self):
        return self.server.server_name
    def get_server_site(self):
        return self.server.server_site
    def get_user(self):
        return self.server.user
    def set_user(self, user):
        self.server.user = user
    def set_server_site(self, cls):
        self.server.server_site = cls       

    # User Features
    def register(self, user: IUserEntity):
        admin = self.get_user()
    
        new_account_info = admin.users.create(
            email = user.email,
            name = user.name,
            password = user.password,
            password_verify = user.password,
            institution = user.institution,
            website = user.website,
            role = user.role
        )
        return new_account_info.id
        
    def delete_user(self, uid):
        self.get_user().users.delete(uid)
        
    def launch(self):
        server_site = sy.orchestra.launch(
            name="my_server",
            reset=True,
            local_db=True,      # 로컬 DB 사용
            port="auto",
        )
        self.set_server_site(server_site)
        
    def login(self, email, password):
        # """
        # email, password로 서버에 로그인하여 user client 획득
        # """
        server_site = self.get_server_site()
        user = server_site.login(email = email, password = password)
        self.set_user(user)
        
    def update_user(self, user: IUserEntity):
        self.get_user().users.update(
            uid=user.uid, 
            name=user.name,
            email=user.email,
            password=user.password,
            institution=user.institution,
            website=user.website,
            role=user.role
        )
        
    # Data Features
    def refresh(self):
        self.get_user().refresh()

    def get_projects(self):
        """
        Returns the list of projects visible to the current user.
        """
        return self.get_user().projects

    def get_requests(self):
        """
        Returns the list of requests across all projects (if admin).
        """
        return self.get_user().requests