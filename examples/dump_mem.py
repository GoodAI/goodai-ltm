from goodai.ltm.mem.auto import AutoTextMemory

# This example dumps the contents of a memory instance

if __name__ == '__main__':
    text_example = """\
    Jake Morales: Hey Archie, what do you think about teaming up with me and Isaac Coax? 
    We could come up with a plan that would distract Lucas Fern.
    Archie: That would be great. Thanks for helping me out."""

    mem = AutoTextMemory.create()
    mem.add_text(text_example, metadata={'foo': 'bar'})

    mem.dump()
