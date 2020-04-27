conda activate tf_lite
cd C:/Users/Lukas/Documents/cd_dash_2/webapp
heroku container:login
heroku container:push web --app convdip
heroku container:release web --app convdip
