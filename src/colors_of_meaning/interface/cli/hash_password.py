import getpass

from colors_of_meaning.infrastructure.security.basic_authentication import hash_password

PROMPT_MESSAGE = "Password to hash: "


def main() -> None:
    entered = getpass.getpass(PROMPT_MESSAGE)
    print(hash_password(entered))


if __name__ == "__main__":
    main()
