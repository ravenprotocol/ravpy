import os

os.environ['RAVENVERSE_HOST'] = "104.197.184.224"
os.environ['RAVENVERSE_PORT'] = "9999"
os.environ['RAVENVERSE_FTP_HOST'] = "104.197.184.224"

from ravpy.distributed.participate import participate
from ravpy.initialize import initialize

if __name__ == '__main__':
    initialize(
        "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjU0ODEyNzExLCJpYXQiOjE2NTQ1OTY3MTEsImp0aSI6IjdkODhiMzNlMzJjZDQ5NzI5N2NhZDM0ZGQ2YWJjNjhkIiwidXNlcl9pZCI6IjEifQ.XHwyRKaDuQkbIhhbyTjx1DTRFmwQwiuXpKQ_IQyLXP8")
    participate()
