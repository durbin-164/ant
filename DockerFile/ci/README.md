#### Note: imagine your terminal is in root user if not use "sudo su"
## docker image generation 

#### Note: One time build docker image and use latter docker load. 

Step1: Install docker in your local pc
Step2: After Install docker in your pc run following command to build ant-ci docker image in docker engine.
NB: need too much time. ( more than an hour )

```
$ cd ant/DockerFile/ci
$ docker build --network=host --tag ant-ci-base .
```
Check docker image in docker engine

```
$ docker images
```
Save docker image in your local pc for later use.

```
$ cd  your desired save location 
$ docker save -o ant-ci-base.tar ant-ci-base:latest
```

### if ant-ci-base is not available in docker engine
then
#### load docker image in docker engine

```
$ cd  your ant-ci-base.tar dir
$ docker load --input ant-ci-base.tar
$ docker images  # to check image load perfectly in docker engine
```
end



## Gitlab Runner

### Install gitlab runner in your local pc

https://docs.gitlab.com/runner/install/

### Register gitlab runner
https://docs.gitlab.com/runner/register/index.html

use executer : docker
docker image: ant-ci-base:latest


### Install --runtime nvidia 
https://stackoverflow.com/questions/59008295/add-nvidia-runtime-to-docker-runtimes

or https://github.com/nvidia/nvidia-container-runtime#daemon-configuration-file

#### After that restart your docker 
```
systemctl restart docker
```


### Edit gitlab-runner config to add network and runner and use your local docker
```
gedit /etc/gitlab-runner/config.toml
```

#### Add this in config: 
runtime = "nvidia"<br/>
network_mode = "host"<br/>
pull_policy = "if-not-present"<br/>

After editing your runner look like:
```
[[runners]]
  name = "name of your runner"
  url = "your url"
  token = "your key"
  executor = "docker"
  [runners.custom_build_dir]
  [runners.cache]
    [runners.cache.s3]
    [runners.cache.gcs]
  [runners.docker]
    tls_verify = false
    image = "ant-ci-base:latest"
    runtime = "nvidia"
    privileged = false
    disable_entrypoint_overwrite = false
    oom_kill_disable = false
    disable_cache = false
    volumes = ["/cache"]
    network_mode = "host"
    shm_size = 0
    pull_policy = "if-not-present"
```

### restart gitlab runner
```
$ gitlab-runner restart
```