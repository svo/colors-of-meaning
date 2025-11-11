source "docker" "arm64" {
  commit      = "true"
  image       = "redocly/redoc"
  run_command = ["-d", "-i", "-t", "--name", "packer-python-sprint-zero-documentation", "{{.Image}}", "/bin/sh"]
  changes = [
    "ENV SPEC_URL ${var.version}.json",
    "CMD [\"sh\", \"/usr/local/bin/docker-run.sh\"]"
  ]
  platform    = "linux/arm64/v8"
}

source "docker" "amd64" {
  commit      = "true"
  image       = "redocly/redoc"
  run_command = ["-d", "-i", "-t", "--name", "packer-python-sprint-zero-documentation", "{{.Image}}", "/bin/sh"]
  changes = [
    "ENV SPEC_URL ${var.version}.json",
    "CMD [\"sh\", \"/usr/local/bin/docker-run.sh\"]"
  ]
  platform    = "linux/amd64"
}

variable "version" {
  type = string
}

build {
  sources = [
    "source.docker.arm64",
    "source.docker.amd64",
  ]

  provisioner "shell" {
    inline = [
      "apk add curl"
    ]
  }

  provisioner "file" {
    source = "${var.version}.json"
    destination = "/usr/share/nginx/html/${var.version}.json"
  }

  post-processors {
    post-processor "docker-tag" {
      repository = "svanosselaer/python-sprint-zero-documentation"
      tags       = ["${source.name}"]
    }
  }
}
