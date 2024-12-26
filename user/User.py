import syft as sy
from data.Data import DataHelper
from.interface.IUserEntity import IUserEntity

class UserEntity:
    def __init__(self,**params):
        self.uid             = params.get('uid')
        self.email           = params.get('email')
        self.name            = params.get('name')
        self.password        = params.get('password')
        self.password_verify = params.get('password_verify', self.password)
        self.institution     = params.get('institution')
        self.website         = params.get('website')
        self.role            = params.get('role')
    # def get_uid(self):
    #     return self.uid
    # def get_name(self):
    #     return self.name
    # def get_email(self):
    #     return self.email
    # def get_password(self):
    #     return self.password
    # def get_institution(self):
    #     return self.institution
    # def get_website(self):
    #     return self.website
    # def get_role(self):
    #     return self.role

    # def set_uid(self, uid): self.uid = uid
    # def set_email(self, email): self.email = email
    # def set_name(self, name): self.name = name
    # def set_password(self, password):
    #     self.password = password
    #     self.password_verify = password
    # def set_institution(self, institution): self.institution = institution
    # def set_website(self, website): self.website = website
    def set_user_by_params(self, **params):
        if 'name' in params:
            self.name = params['name']
        if 'institution' in params:
            self.institution = params['institution']
        if 'website' in params:
            self.website = params['website']
        if 'email' in params:
            self.email = params['email']
        if 'password' in params:
            self.password = params['password']
            self.password_verify = params['password']
    
class UserRepository:
    def __init__(self, server_helper: DataHelper):
        self.server_helper = server_helper

    # def get_user_by_id(self, user_id: int) -> DataEntity:
    #     # 여기서는 실제 DB에서 user_id로 유저 정보를 검색하는 로직
    #     # 예제를 위해 가짜 데이터 반환
    #     # 실제로는 SQL 쿼리 실행 혹은 ORM 사용
        
    #     return UserEntity(user_id=user_id, name="John Doe", email="john@example.com")

    def register(self, user: UserEntity):
        iuser = IUserEntity(
            uid=user.uid,
            email=user.email,
            name=user.name,
            password=user.password,
            institution=user.institution,
            website=user.website,
            role=user.role
        )
        new_uid = self.server_helper.register(iuser)
        user.uid = new_uid

    def delete_user(self, user: UserEntity):
        self.server_helper.delete_user(user.uid)
        
    def login(self, email, password):
        self.server_helper.login(email=email, password=password)
    
    def update_user(self, user: UserEntity):
        iuser = IUserEntity(
            uid=user.uid,
            email=user.email,
            name=user.name,
            password=user.password,
            institution=user.institution,
            website=user.website,
            role=user.role
        )
        # server_helper를 통해 사용자 업데이트
        self.server_helper.update_user(iuser)

        # self.get_user().users.update(
        #     uid=user.uid, 
        #     name=user.name,
        #     email=user.email,
        #     password=user.password,
        #     institution=user.institution,
        #     website=user.website,
        #     role=user.role
        # )