"""Explicitly-labeled SYNTHETIC demo transcripts.

Everything here is invented from templates (companies, people, dates, competitors). It exists only so the
pipeline and dashboard can be exercised when no real transcripts are available. Findings computed on it
are NOT company findings. ``data_origin`` is always "synthetic".

Template syntax: ``<<a|b|c>>`` picks one option at random; ``{slot}`` is filled from SLOTS.
"""
import hashlib
import json
import random
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

COMPANIES = ["Acme Freight", "Brightwave Labs", "Cobalt Retail", "Dunmore Health", "Evergreen Foods", "Fathom Media",
             "Granite Logistics", "Helix Studios", "Ironbridge Capital", "Juniper Energy", "Kestrel Travel", "Lumen Schools",
             "Meridian Bank", "Northpeak Gear", "Orchid Pharma", "Pinecrest Realty", "Quartz Telecom", "Redwood Legal"]
SLOTS = {
    "n": ["12", "30", "45", "80", "150"],
    "idp": ["Okta", "Azure AD", "Ping Identity"],
    "svc": ["the reporting API", "the ingestion service", "the dashboard", "the webhook service"],
    "feature": ["CSV export", "webhooks for ticket events", "role-based dashboards", "a Salesforce integration", "scheduled reports", "SAML group mapping"],
    "competitor": ["Fluxio", "Datanest", "Quillbase"],
}
PEOPLE = ["Alex", "Sam", "Jordan", "Taylor", "Casey", "Riley", "Morgan", "Jamie", "Avery", "Drew", "Quinn", "Reese"]

# (speaker, template) lines per scenario. n values sum to 100.
SCENARIOS = [
    dict(key="sso_login", call_type="support", n=7, lines=[
        ("Agent", "Thanks for contacting support, this is <<Maya|Dan|Priya|Luis>>. How can I help?"),
        ("Customer", "Our users can't log in through SSO since we changed {idp} <<last night|last week>>. <<It is blocking about {n} people.|Half our team is locked out.>>"),
        ("Agent", "I'm sorry about that. Can you share the SAML error you see?"),
        ("Customer", "It says <<invalid assertion signature|SAML callback failed with 403>>. <<This is really frustrating, we have a deadline today.|Not sure what changed on our side.>>"),
        ("Agent", "It looks like the signing certificate on the identity provider side doesn't match the one in your tenant. <<I've escalated to our identity team.|Let's update the certificate together.>>"),
        ("Customer", "<<Okay, please be quick, this is unacceptable for a paid plan.|Okay, that makes sense. Let's try it.>>"),
        ("Agent", "I'll send the steps by email and follow up in an hour.")]),
    dict(key="mfa_lockout", call_type="support", n=4, lines=[
        ("Agent", "Support here, how can I help today?"),
        ("Customer", "My MFA codes stopped working after I got a new phone. I'm locked out of my admin account."),
        ("Agent", "No problem, I can reset your MFA device after verifying your identity."),
        ("Customer", "Great, thanks. <<I verified by email already.|That would be perfect.>>"),
        ("Agent", "Done. Please log in and re-enroll a device. I'll stay on the line."),
        ("Customer", "It works now. I appreciate the quick help.")]),
    dict(key="double_charge", call_type="support", n=6, lines=[
        ("Agent", "Hello, thanks for calling billing support."),
        ("Customer", "We were charged twice for the <<March|April|May>> invoice. <<That is a lot of money for a small team.|I'm worried about our cash flow.>>"),
        ("Agent", "I see two payments on the credit card ending 4417. I'm sorry about the duplicate charge."),
        ("Customer", "<<Can you refund it today? This is the second time this has happened.|Please refund one of them, I'd appreciate it.>>"),
        ("Agent", "I've issued the refund. It takes five to seven business days. I'll send a confirmation receipt."),
        ("Customer", "<<Okay, thank you. Please make sure it doesn't happen again.|Thanks, that's helpful.>>")]),
    dict(key="invoice_tax", call_type="support", n=4, lines=[
        ("Agent", "Support, how can I help?"),
        ("Customer", "The invoice shows the wrong tax rate for our region. Finance flagged it and they are annoyed."),
        ("Agent", "Thanks for flagging. I'll check the tax settings on your billing profile."),
        ("Customer", "Please correct it and reissue the invoice. Everything else about the service is good."),
        ("Agent", "I will reissue it today and send a corrected receipt.")]),
    dict(key="api_latency", call_type="support", n=6, lines=[
        ("Agent", "Support here. What's going on?"),
        ("Customer", "Calls to {svc} are very slow, p95 latency is over <<8|12|20>> seconds and we see timeouts."),
        ("Agent", "Thanks. Are the timeouts only on large queries, or on everything?"),
        ("Customer", "<<Everything since Monday. It is hurting our customers and I'm disappointed.|Mostly large exports. It's getting painful.>>"),
        ("Agent", "I can see elevated response time on your region. I'll open a performance ticket and follow up."),
        ("Customer", "<<Please escalate this, we are losing time every hour.|Okay, please keep me posted.>>")]),
    dict(key="outage_report", call_type="support", n=7, lines=[
        ("Agent", "Support, this is <<Maya|Dan|Priya|Luis>>."),
        ("Customer", "{svc} is down for us, we get 503 errors. This outage is costing us money and it's unacceptable."),
        ("Agent", "I'm sorry. I can confirm an incident and the status page is being updated."),
        ("Customer", "<<This is the second outage this quarter. If it keeps happening we will cancel and look at {competitor}.|Our whole operations team is stuck. I'm angry, please get an executive on the line.>>"),
        ("Agent", "I've escalated this as sev1 and engineering is working on it. I'll update you every thirty minutes."),
        ("Customer", "<<Fine, but we need a root cause in writing.|We are evaluating other vendors if this isn't fixed.>>")]),
    dict(key="feature_gap_support", call_type="support", n=6, lines=[
        ("Agent", "Hi, thanks for contacting support."),
        ("Customer", "Is there a way to get {feature}? We need a way to automate this and it's missing today."),
        ("Agent", "That isn't available right now, but I can log a feature request for you."),
        ("Customer", "<<That would be great, it would be nice if it landed soon.|Okay. We'd love to see it on the roadmap.>>"),
        ("Agent", "I'll file it with product and send you the reference.")]),
    dict(key="renewal_at_risk", call_type="external", n=8, lines=[
        ("Account Manager", "Thanks for making time. I wanted to talk about your renewal in <<two|three>> months."),
        ("Client", "Honestly we're on the fence about the renewal. Pricing went up and the budget is tight this year."),
        ("Account Manager", "I understand. What would help you decide?"),
        ("Client", "<<We are evaluating {competitor} and comparing prices. The outages last quarter worried us.|Leadership asked us to look at alternatives. We may switch if we can't get a discount.>>"),
        ("Account Manager", "I hear you. Let me see what we can do on price and reliability commitments."),
        ("Client", "<<Please send a proposal, but I'm not optimistic.|Okay, send something. We haven't decided yet.>>")]),
    dict(key="renewal_happy", call_type="external", n=5, lines=[
        ("Account Manager", "Great to see you, ready to talk renewal?"),
        ("Client", "Yes, we're happy with the platform and plan to renew. Adoption has been excellent."),
        ("Account Manager", "Wonderful. Would you like to expand to more seats?"),
        ("Client", "<<Absolutely, we want to add {n} seats and we're excited about the new reports.|Yes, the team loves it and the support has been great.>>"),
        ("Account Manager", "I'll send the renewal quote and an expansion proposal this week.")]),
    dict(key="onboarding_adoption", call_type="external", n=7, lines=[
        ("Account Manager", "Let's review adoption since the kickoff."),
        ("Client", "Usage is low. Only about a quarter of our seats are logging in weekly and the training sessions were poorly attended."),
        ("Account Manager", "Thanks for being open. Do you have an internal champion?"),
        ("Client", "<<Not really, our champion left. We are worried we're not getting value.|Yes, but they are overloaded. We need help with onboarding.>>"),
        ("Account Manager", "We can run a new enablement workshop and activate the remaining teams."),
        ("Client", "<<That would be helpful, let's schedule it.|Okay, I'll send names of who should attend.>>")]),
    dict(key="feature_feedback", call_type="external", n=9, lines=[
        ("Account Manager", "I'd love your product feedback before the roadmap review."),
        ("Client", "The product is good overall. We need {feature}. Do you support that on the roadmap?"),
        ("Account Manager", "I'll note it. How important is it for your workflow?"),
        ("Client", "<<It is missing and it's a blocker for two of our teams, a little disappointing.|It would be great if it shipped this year, otherwise we need a workaround.>>"),
        ("Account Manager", "Understood. I will follow up with product and share an update.")]),
    dict(key="pricing_negotiation", call_type="external", n=6, lines=[
        ("Account Manager", "Thanks for the call. I have the pricing proposal."),
        ("Client", "The price increase is too expensive for us. Finance cut our budget and we need a discount."),
        ("Account Manager", "We can offer <<10|15>> percent for a two year contract."),
        ("Client", "<<That helps but it is still above what {competitor} quoted us.|Okay, that is closer. Let me take it to finance.>>"),
        ("Account Manager", "I'll send an updated quote and we can talk again <<Friday|next week>>.")]),
    dict(key="release_delay", call_type="internal", n=7, lines=[
        ("Eng Manager", "Let's review the release status. Where do we stand on the <<2.4|2.5|3.0>> release?"),
        ("Engineer A", "We are delayed. The integration tests are flaky and the staging deploy failed twice."),
        ("Engineer B", "QA is blocked waiting on the build. I think we need to slip the release by a week."),
        ("Eng Manager", "<<Okay, let's freeze scope and I'll tell product.|That's frustrating but fine. I'll update the release train.>>"),
        ("Engineer A", "I'll fix the flaky tests today and we will redeploy to staging tomorrow."),
        ("Engineer B", "Action item: someone should own the rollback plan before we ship.")]),
    dict(key="incident_postmortem", call_type="internal", n=6, lines=[
        ("Eng Manager", "Thanks everyone for joining the incident review for last week's SSO and login outage."),
        ("Engineer A", "Root cause was an expired signing certificate in the auth service, which caused SAML callback failures."),
        ("Engineer B", "Customers saw 403 errors and some saw 503 on {svc}. We rolled back the config change within an hour."),
        ("Eng Manager", "What's still unresolved?"),
        ("Engineer A", "<<Monitoring didn't alert us. That's a gap and still unresolved.|We have no automated certificate renewal yet, it is a dependency on the platform team.>>"),
        ("Engineer B", "I'll add the alert and we will write the follow-up tasks today. Let's review in two weeks.")]),
    dict(key="blocker_dependency", call_type="internal", n=6, lines=[
        ("Eng Manager", "What's blocking the latency work?"),
        ("Engineer A", "We are blocked on the platform team for the new cache layer. They have no capacity until next quarter."),
        ("Engineer B", "The backlog of tech debt in the query service is also slowing us down, and the review bottleneck is real."),
        ("Eng Manager", "<<I'll escalate the dependency to the director.|Let's pull one engineer from the roadmap to unblock this.>>"),
        ("Engineer A", "That helps. I will write the design doc and send it for review.")]),
    dict(key="escalation_planning", call_type="internal", n=6, lines=[
        ("Support Lead", "We have several customers escalating about SSO login failures and one threatening to cancel."),
        ("Eng Manager", "How many accounts are affected?"),
        ("Support Lead", "About <<five|eight>> that we know of. Two said they're evaluating competitors if it isn't fixed."),
        ("Engineer A", "We can prioritize a fix for the SAML certificate handling, but it competes with the release."),
        ("Eng Manager", "<<Let's make it the top priority. I'll update the release plan and brief customer success.|Okay, I'll take the action item to set up a daily sync until it's resolved.>>"),
        ("Support Lead", "Thanks, I'll send the affected account list today.")]),
]

GROUND_TRUTH = {
    "sso_login": ("Authentication & Access", "Mixed"),
    "mfa_lockout": ("Authentication & Access", "Positive"),
    "double_charge": ("Billing & Payments", "Mixed"),
    "invoice_tax": ("Billing & Payments", "Mixed"),
    "api_latency": ("Performance & Latency", "Negative"),
    "outage_report": ("Reliability & Outages", "Negative"),
    "feature_gap_support": ("Product Features & Feedback", "Mixed"),
    "renewal_at_risk": ("Renewals & Pricing", "Negative"),
    "renewal_happy": ("Renewals & Pricing", "Positive"),
    "onboarding_adoption": ("Adoption & Onboarding", "Mixed"),
    "feature_feedback": ("Product Features & Feedback", "Mixed"),
    "pricing_negotiation": ("Renewals & Pricing", "Mixed"),
    "release_delay": ("Deployment & Release Management", "Mixed"),
    "incident_postmortem": ("Reliability & Outages", "Neutral"),
    "blocker_dependency": ("Engineering Blockers", "Negative"),
    "escalation_planning": ("Engineering Blockers", "Mixed"),
}

FOLLOWUPS = {
    "sso_login": [
        ("Customer", "I can test with one affected user before we roll this out to the whole team."),
        ("Agent", "Please confirm whether the SAML login succeeds with the updated certificate."),
        ("Customer", "I'll test with our admin account now and send you the result."),
    ],
    "mfa_lockout": [
        ("Agent", "Before we wrap up, can you confirm the backup codes are available?"),
        ("Customer", "Yes, I have them stored and will update my backup device."),
        ("Agent", "Great. I will add the recovery steps to the support case."),
        ("Customer", "Confirmed, I can reach the admin console now."),
    ],
    "double_charge": [
        ("Agent", "Could you confirm which invoice number should remain on the account?"),
        ("Customer", "Keep the original invoice and reverse the second payment."),
        ("Agent", "Understood. I have added that instruction to the refund request."),
        ("Customer", "Thanks, I will watch for the confirmation email."),
    ],
    "invoice_tax": [
        ("Customer", "Can you confirm whether finance needs to submit a new tax form?"),
        ("Agent", "The current profile has the right address; I will verify the tax code with billing."),
        ("Customer", "Please let me know if you need anything else from our finance team."),
        ("Agent", "I will include the corrected invoice and the reason for the change in my update."),
        ("Customer", "All right. We will hold payment until the corrected copy arrives."),
    ],
    "api_latency": [
        ("Customer", "Can you share an estimated time for the performance investigation?"),
        ("Agent", "I will attach your request IDs and the affected region to the ticket."),
        ("Customer", "Please keep the case open until the timeouts stop."),
        ("Agent", "Agreed. I will send the next update even if the investigation is still underway."),
    ],
    "outage_report": [
        ("Customer", "Our team is checking the status page, but we need an update for our operations lead."),
        ("Agent", "I will include the current incident status and the next update time in the case."),
        ("Customer", "Please send the incident summary once service is stable."),
        ("Agent", "Understood. I will follow up with the incident reference and any confirmed details."),
    ],
    "feature_gap_support": [
        ("Customer", "The team currently exports each report manually, which takes several hours."),
        ("Agent", "I will include that workflow impact with the request for the product team."),
        ("Customer", "A status update would help us decide whether to build a temporary workaround."),
        ("Agent", "I can share the request reference, but I cannot promise a delivery date."),
        ("Customer", "That is fair. Please send the reference when it is available."),
    ],
    "renewal_at_risk": [
        ("Client", "Reliability and the total renewal cost are the two issues leadership wants addressed."),
        ("Account Manager", "I will separate the pricing proposal from the reliability follow-up."),
        ("Client", "Please include the renewal deadline so we can review both items internally."),
        ("Account Manager", "I will send the proposal and schedule a check-in before that deadline."),
    ],
    "renewal_happy": [
        ("Client", "Our operations group has been using the new reports in its weekly review."),
        ("Account Manager", "That is helpful context for the seat expansion estimate."),
        ("Client", "Please keep the renewal terms and added seats on the same proposal."),
        ("Account Manager", "I will send one draft with the current and expanded options."),
        ("Client", "Thanks. I will route it to procurement for approval."),
    ],
    "onboarding_adoption": [
        ("Client", "The new users will need a short session focused on their daily workflow."),
        ("Account Manager", "I will tailor the workshop and send a few times for next week."),
        ("Client", "Please include the team leads so they can reinforce the process afterward."),
        ("Account Manager", "I will add them to the invite and share the setup guide."),
    ],
    "feature_feedback": [
        ("Client", "The missing capability adds a manual step to our monthly reconciliation."),
        ("Account Manager", "I will capture that workflow and pass it to the product review."),
        ("Client", "A rough timing estimate would help us plan around the current limitation."),
        ("Account Manager", "I can request an update, but the roadmap team makes the timing decision."),
        ("Client", "Understood. Please share whatever the team can confirm."),
    ],
    "pricing_negotiation": [
        ("Client", "We need the renewal total to stay within the budget approved this quarter."),
        ("Account Manager", "I will show the revised term and discount as separate line items."),
        ("Client", "Please make the expiration date for the quote clear."),
        ("Account Manager", "I will include that and send the proposal for your finance review."),
        ("Client", "Thank you. I will bring it to our next budget meeting."),
    ],
    "release_delay": [
        ("Eng Manager", "What dependency should we raise first if the staging build fails again?"),
        ("Engineer A", "The test environment is the immediate dependency; I will attach the failing job output."),
        ("Engineer B", "I will update the release ticket with the rollback owner and revised date."),
        ("Eng Manager", "Let's review the build status in tomorrow's stand-up."),
    ],
    "incident_postmortem": [
        ("Eng Manager", "Who owns the certificate renewal work and the missing alert?"),
        ("Engineer A", "I will own the alert; the platform team needs to confirm the renewal change."),
        ("Engineer B", "I will link both tasks to the incident review record."),
        ("Eng Manager", "Please bring the verification results to the follow-up review."),
    ],
    "blocker_dependency": [
        ("Eng Manager", "Can we split the cache work so the team can make progress before next quarter?"),
        ("Engineer A", "I can document the interface, but the platform team still owns the deployment dependency."),
        ("Engineer B", "I will write down the review bottleneck and estimate the work we can do in parallel."),
        ("Eng Manager", "Let's review both owners and dates at the next planning meeting."),
        ("Engineer A", "I will send the design outline before then."),
    ],
    "escalation_planning": [
        ("Eng Manager", "What evidence should support use when customers ask for an update?"),
        ("Support Lead", "We can share the confirmed workaround and the next checkpoint, not an unverified ETA."),
        ("Eng Manager", "Agreed. I will send that guidance to the account teams today."),
        ("Support Lead", "I will update the affected accounts after the daily sync."),
    ],
}


def _fill(template: str, rng: random.Random) -> str:
    text = re.sub(r"<<(.*?)>>", lambda m: rng.choice(m.group(1).split("|")), template)
    return re.sub(r"\{(\w+)\}", lambda m: rng.choice(SLOTS[m.group(1)]), text)


def _generate(n_total: int = 100, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Generate clean transcripts and separate scenario-designed evaluation labels."""
    if n_total < 1:
        raise ValueError("n_total must be at least 1")
    rng = random.Random(seed)
    weights = [s["n"] / 100 for s in SCENARIOS]
    counts = [int(n_total * w) for w in weights]
    for i in range(n_total - sum(counts)):
        counts[i % len(counts)] += 1
    start = date(2026, 4, 1)
    seen: set[str] = set()
    records: list[dict] = []
    labels: list[dict] = []
    for scen, count in zip(SCENARIOS, counts):
        for _ in range(count):
            for _ in range(200):
                company = rng.choice(COMPANIES)
                lines = [(who, _fill(t, rng)) for who, t in scen["lines"]]
                if scen["call_type"] == "internal":
                    lines[0] = (lines[0][0], f"(Sprint {rng.randint(20, 60)}) {lines[0][1]}")
                else:
                    intro = f"Hi, this is {rng.choice(PEOPLE)} from {company}. "
                    lines[1] = (lines[1][0], intro + lines[1][1])
                lines.extend((who, _fill(text, rng)) for who, text in FOLLOWUPS[scen["key"]])
                if len(lines) != 10:
                    raise ValueError(f"{scen['key']} must generate exactly 10 speaker turns, got {len(lines)}")
                call_date = start + timedelta(days=rng.randint(0, 183))
                started_at = datetime.combine(call_date, time(rng.randint(8, 16), rng.choice([0, 15, 30, 45])),
                                              tzinfo=timezone.utc)
                turn_at = started_at
                timestamped_lines = []
                for who, utterance in lines:
                    timestamped_lines.append(f"[{turn_at:%H:%M}] {who}: {utterance}")
                    turn_at += timedelta(minutes=rng.randint(1, 4))
                text = "\n".join(timestamped_lines)
                if text not in seen:
                    break
            if text in seen:
                raise RuntimeError(f"Unable to generate a unique transcript for scenario {scen['key']}")
            seen.add(text)
            topic_label, sentiment_label = GROUND_TRUTH[scen["key"]]
            records.append({
                "transcript_id": "", "call_type": scen["call_type"],
                "call_date": started_at.isoformat(timespec="minutes"),
                "customer_id": "" if scen["call_type"] == "internal" else f"CUST-{COMPANIES.index(company) + 1:02d}",
                "transcript_text": text, "data_origin": "synthetic"})
            labels.append({
                "transcript_id": "", "call_type": scen["call_type"], "topic_label": topic_label,
                "sentiment_label": sentiment_label, "data_origin": "synthetic",
                "text_hash": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]})
    paired = list(zip(records, labels))
    rng.shuffle(paired)
    records = [record for record, _ in paired]
    labels = [label for _, label in paired]
    for i, (r, label) in enumerate(zip(records, labels), 1):
        r["transcript_id"] = f"SYN-{i:03d}"
        label["transcript_id"] = r["transcript_id"]
    return records, labels


def generate(n_total: int = 100, seed: int = 42) -> list[dict]:
    """Generate exactly ``n_total`` synthetic transcripts, deterministic for a given seed."""
    return _generate(n_total, seed)[0]


def write_synthetic(out_dir, n_total: int = 100, seed: int = 42) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs, labels = _generate(n_total, seed)
    path = out_dir / "transcripts.json"
    path.write_text(json.dumps({"_note": "SYNTHETIC demo data. Not real calls.", "transcripts": recs}, indent=2),
                    encoding="utf-8")
    pd.DataFrame(recs).to_csv(out_dir / "transcripts.csv", index=False)
    (out_dir / "ground_truth.json").write_text(
        json.dumps({"_note": "Scenario-designed synthetic labels; not human-reviewed.", "labels": labels}, indent=2),
        encoding="utf-8")
    pd.DataFrame(labels).to_csv(out_dir / "ground_truth.csv", index=False)
    return path
