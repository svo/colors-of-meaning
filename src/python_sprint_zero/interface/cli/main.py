import click


@click.command()
@click.option("--message", default="coconuts", help="Message to print")
def run(message: str) -> None:
    print(message)
