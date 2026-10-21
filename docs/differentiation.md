# Why SlothCTF is different from other CTF generators

## Existing project

The repository at https://github.com/ayeshasq/ctf-generator focuses on a polished, user-facing CTF generation experience:

- challenge categories
- web UI and frontend
- scoring and flags
- tutorial/story-based challenge generation
- broad educational use-case

This is a strong project, but it is mainly oriented toward challenge consumption and content generation for users.

## SlothCTF direction

SlothCTF takes a different path:

- local-first execution
- open-weight model usage
- artifact generation instead of only story generation
- build and validation loop
- source code + binary + solver + exports
- privacy-preserving and offline-friendly workflow

## Positioning

We are not trying to be a duplicate of a general CTF generator.

We are building a local challenge authoring workflow for security educators and challenge creators, with the open-source AI model doing the repetitive part while keeping the data local and controllable.

This makes SlothCTF a better fit for the dev.to Slow and Steady challenge theme, because it focuses on automation, privacy, and local-first AI tooling.
