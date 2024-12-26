class IUserEntity:
    def __init__(self,**params):
        self.uid             = params.get('uid')
        self.email           = params.get('email')
        self.name            = params.get('name')
        self.password        = params.get('password')
        self.password_verify = self.password
        self.institution     = params.get('institution')
        self.website         = params.get('website')
        self.role            = params.get('role')