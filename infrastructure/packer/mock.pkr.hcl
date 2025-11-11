source "docker" "arm64" {
  commit      = "true"
  image       = "node:18-alpine"
  run_command = ["-d", "-i", "-t", "--name", "packer-colors-of-meaning-mock-arm64", "--entrypoint", "/bin/sh", "{{.Image}}"]
  changes = [
    "WORKDIR /app",
    "ENTRYPOINT [\"prism\"]",
    "CMD [\"mock\", \"-h\", \"0.0.0.0\", \"-p\", \"5000\", \"/app/schema.json\"]"
  ]
  platform    = "linux/arm64/v8"
}

source "docker" "amd64" {
  commit      = "true"
  image       = "node:18-alpine"
  run_command = ["-d", "-i", "-t", "--name", "packer-colors-of-meaning-mock-amd64", "--entrypoint", "/bin/sh", "{{.Image}}"]
  changes = [
    "WORKDIR /app",
    "ENTRYPOINT [\"prism\"]",
    "CMD [\"mock\", \"-h\", \"0.0.0.0\", \"-p\", \"5000\", \"/app/schema.json\", \"--cors\"]"
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
      "mkdir /app",
      "npm i -g @stoplight/prism-cli@5.12.0",
      "apk add curl"
    ]
  }

  provisioner "file" {
    source = "${var.version}.json"
    destination = "/app/schema.json"
  }

  post-processors {
    post-processor "docker-tag" {
      repository = "svanosselaer/colors-of-meaning-mock"
      tags       = ["${source.name}"]
    }
  }
}
