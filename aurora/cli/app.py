import typer
from aurora.cli.train_app import train_app
from aurora.cli.gen_app import gen_app
from aurora.cli.plot_app import plot_app

app = typer.Typer()
app.add_typer(train_app, name="train")
app.add_typer(gen_app, name="gen")
app.add_typer(plot_app, name="plot")

def main():
    app()