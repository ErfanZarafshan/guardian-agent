# 🎬 Guardian Agent — Demo Video Script

**Target length:** 5 minutes (the rubric allows up to 5 min for groups
of ≤2). Aim for 4:30 to give buffer.

**Recording tool:** macOS QuickTime (Cmd+Shift+5 → Record Selected
Portion → pick the entire screen). Audio: built-in mic is fine; speak
slowly and close to the laptop.

**Setup before you hit record:**
- Three browser tabs ready in this order, left to right:
  1. The deployed app: <https://guardianagent.streamlit.app/>
  2. Same URL in a second tab
  3. Your GitHub repo: <https://github.com/ErfanZarafshan/guardian-agent>
- One terminal window open, venv activated, in the project directory
- Close Slack, mail, anything that might pop a notification
- On Tab 1: load the **Baseline** profile and click Save
- On Tab 2: load the **Vulnerable** profile and click Save
- Have your LLM API key copied if you plan to demo that part

---

## Script

### [0:00 — 0:30] The problem (30s)

> "Hi, I'm Erfan Zarafshan. This is Guardian Agent — a CSC 4444G project
> by myself and Maby Gavilan Abanto."
>
> *(Switch to GitHub repo tab — show README briefly)*
>
> "Existing weather emergency alerts have a fundamental limitation:
> they're broadcast. Every cell phone in a polygon gets the same text,
> regardless of who the recipient actually is. A 78-year-old on the
> ground floor of a flood-prone apartment with no car gets the same
> message as a 25-year-old on a third floor with a 4WD truck. The
> right response is materially different. Guardian Agent fixes that
> with personalized probabilistic reasoning."

### [0:30 — 1:30] The architecture (60s)

> *(Switch to deployed app, click "About" page in the left sidebar)*
>
> "The architecture is a goal-based agent in the AIMA Chapter 2 sense.
> Three inputs — live weather from NWS and OpenWeatherMap, an ML threat
> classifier trained on 106,000 NOAA Storm Events, and a structured
> user profile — feed into a 10-node Bayesian network. Inference is
> exact via Variable Elimination, and produces a posterior over four
> risk levels: Low, Moderate, High, and Critical."
>
> *(Scroll down on About page to show the methodology block)*
>
> "A deterministic rule planner then maps the posterior into actions
> across three channels: console, SMS via Twilio, and a smart-home mock
> that an Alexa skill could implement. Every action carries a written
> rationale, so the system is auditable end to end. The classifier sits
> at ROC-AUC 0.76 on held-out data, which is the honest ceiling for a
> calendar-and-location-only feature set."

### [1:30 — 2:15] Live weather perception (45s)

> *(Switch to Tab 1 — Baseline profile already saved. Click "Live
> Weather" in sidebar.)*
>
> "First, the perception layer. I click 'Fetch live weather now' and
> the agent calls both NWS and OpenWeatherMap for my address in Baton
> Rouge."
>
> *(Click the button)*
>
> "We get back current temperature, wind, humidity, and any active NWS
> alerts. Right now there's nothing severe — that's the truthful state
> of the world."

### [2:15 — 3:00] Risk assessment (45s)

> *(Click "Risk Assessment" in sidebar)*
>
> "Now the reasoning step. The Bayesian engine takes everything we just
> saw, plus the profile vulnerability factors, and produces this
> posterior."
>
> *(Point at the bar chart)*
>
> "Low at 48%, Moderate at 32%, High at 14%, Critical at 6% — the
> agent agrees with the perception layer that conditions are calm, and
> the planner produces only a log entry. No SMS, no alarms, no
> spurious notifications. This is the right behavior for a clear day."

### [3:00 — 4:00] The personalization punchline (60s)

> *(Click "Scenario Simulator" in sidebar. From the dropdown on the
> left, pick "Tornado Warning")*
>
> "Now the killer feature. The Scenario Simulator lets us inject any
> hypothetical hazard. I'll pick Tornado Warning."
>
> *(Bars instantly redraw to dominantly Critical)*
>
> "Critical jumps to 48% for our baseline user, the planner switches
> on shelter-in-place, sounds the alarm, and queues an SMS to the
> emergency contact. Now watch what happens for a different user."
>
> *(Switch to Tab 2 — Vulnerable profile. Navigate to Scenario
> Simulator, pick Tornado Warning again.)*
>
> "Same exact scenario. Same wind, same alert. Different person — Sam
> lives on the ground floor, no vehicle, mobility limited. Notice the
> Critical bar is now at 61% — 13 points higher. The planner generates
> a longer action list, with two contacts paged instead of one. Same
> hazard, two materially different responses. That is the entire
> point of personalized probabilistic reasoning."

### [4:00 — 4:30] LLM explainer (30s, optional)

> *(If you have an API key — go to Risk Assessment, click "Explain in
> plain English")*
>
> "We also integrated an optional LLM explainer. It rephrases the
> posterior and the planner's actions into conversational guidance.
> Critically, the LLM cannot change recommendations — it only
> rephrases what the deterministic system already decided. That keeps
> the safety-critical reasoning grounded."

### [4:30 — 5:00] Wrap (30s)

> "The full system is 239 tests passing across nine test files,
> deployed live at guardianagent.streamlit.app, and on GitHub at
> github.com/ErfanZarafshan/guardian-agent. The full report is in the
> repo. Thanks for watching."

---

## Things to remember while recording

- **Speak slower than you think you need to.** A 5-min script feels
  rushed at normal speed.
- **Don't apologize on camera.** If you fumble, just keep going — you
  can re-record once.
- **Cursor matters.** Move it deliberately to whatever you're talking
  about. Don't wave it around.
- **Mute notifications.** Cmd+Option+D doesn't always silence Slack
  popups; quit those apps entirely.
- **First take won't be perfect — that's fine.** Plan for two takes.

## After recording

1. Trim leading/trailing silence in QuickTime: `Edit → Trim`
2. Save as `.mov` — that's the native QuickTime format
3. If file is big, compress with HandBrake (free) to MP4 ~50 MB
4. Upload to YouTube as **Unlisted** (so only people with the link can
   see it) and put the link in your project's README and the report

That's it. You've now got a video that proves the project works for the
grader, without them needing to install anything.
