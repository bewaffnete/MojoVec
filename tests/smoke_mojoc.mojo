"""
Smoke test to verify that the compiled mojovec.mojoc package can be imported
and used successfully by a downstream client.
"""
from mojovec import Client

def main():
    var client = Client()
    print("Successfully imported and instantiated Client from mojovec.mojoc!")
