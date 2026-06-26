"""Extension point for custom OptForce variants.

Keep the core stages (fva, must_sets, bilevel, optforce) pure. Put new MUST-set
rules, alternative inner objectives, measured-flux integration, regularisation, etc.
here, composing the core primitives rather than editing them.
"""
