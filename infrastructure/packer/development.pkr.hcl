source "docker" "arm64" {
  changes     = ["EXPOSE 22", "CMD [\"/usr/sbin/sshd\", \"-D\"]"]
  commit      = "true"
  image       = "dockette/vagrant:debian-12"
  run_command = ["-d", "-i", "-t", "-v", "/var/run/docker.sock:/var/run/docker.sock", "--name", "packer-python-sprint-zero-development-arm64", "{{.Image}}", "/bin/bash"]
  platform    = "linux/arm64/v8"
}

source "docker" "amd64" {
  changes     = ["EXPOSE 22", "CMD [\"/usr/sbin/sshd\", \"-D\"]"]
  commit      = "true"
  image       = "dockette/vagrant:debian-12"
  run_command = ["-d", "-i", "-t", "-v", "/var/run/docker.sock:/var/run/docker.sock", "--name", "packer-python-sprint-zero-development-amd64", "{{.Image}}", "/bin/bash"]
  platform    = "linux/amd64"
}

build {
  sources = [
    "source.docker.arm64",
    "source.docker.amd64",
  ]

  provisioner "ansible" {
    extra_arguments = ["--extra-vars", "ansible_host=packer-python-sprint-zero-development-${source.name} ansible_connection=docker"]
    playbook_file   = "infrastructure/ansible/playbook-development.yml"
    user            = "vagrant"
  }

  post-processor "docker-tag" {
    repository = "svanosselaer/python-sprint-zero-development"
    tags       = ["${source.name}"]
  }
}
