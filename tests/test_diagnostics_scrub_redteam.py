"""One synthetic case per observed leak or destroyed diagnostic fragment."""
import re

import pytest

from app2.diagnostics.scrub import Scrubber
from data_scrub_corpus import CASES, CONTEXT, PRESERVE


@pytest.mark.parametrize("name,source,secrets,keep", CASES, ids=[c[0] for c in CASES])
def test_identifying_fragments_are_removed_without_losing_diagnostics(name, source, secrets, keep):
    output = Scrubber(CONTEXT).scrub(source)
    assert all(secret not in output for secret in secrets), (name, output)
    assert all(fragment in output for fragment in keep), (name, output)


@pytest.mark.parametrize("name,source,keep", PRESERVE, ids=[c[0] for c in PRESERVE])
def test_versions_gpu_timings_stage_names_and_numbers_survive(name, source, keep):
    output = Scrubber(CONTEXT).scrub(source)
    assert all(fragment in output for fragment in keep), (name, output)


@pytest.mark.parametrize("name,source,secrets,keep", CASES, ids=[c[0] for c in CASES])
def test_scrub_is_idempotent_for_every_adversarial_case(name, source, secrets, keep):
    scrubber = Scrubber(CONTEXT)
    once = scrubber.scrub(source)
    assert scrubber.scrub(once) == once, name
    assert Scrubber(CONTEXT).scrub(once) == once, name


def test_known_context_literals_never_survive_in_corpus():
    output = "\n".join(Scrubber(CONTEXT).scrub(source) for _name, source, _secrets, _keep in CASES)
    literals = [CONTEXT["home"], CONTEXT["user"], CONTEXT["host"]]
    literals += CONTEXT["roots"] + CONTEXT["projects"] + CONTEXT["extra"]
    literals += ["ProjectLive09282026192332"]
    for literal in literals:
        assert not re.search(re.escape(literal), output, re.I), literal


def test_aliases_stay_stable_and_different_inputs_get_different_aliases():
    scrubber = Scrubber(CONTEXT)
    first = scrubber.scrub("ProjectLive09282026192332 09282026222119 09282026222119")
    second = scrubber.scrub("ProjectLive09282026192332 09292026010758")
    assert first == "Project-A side-1 side-1"
    assert second == "Project-A side-2"


def test_combined_report_stays_idempotent():
    source = "\n".join(line for _name, line, _secrets, _keep in CASES)
    scrubber = Scrubber(CONTEXT)
    once = scrubber.scrub(source)
    assert scrubber.scrub(once) == once
    assert Scrubber(CONTEXT).scrub(once) == once
