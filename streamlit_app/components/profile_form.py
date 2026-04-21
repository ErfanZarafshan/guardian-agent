"""Profile-builder UI: interactive form to construct a UserProfile.

Used on the Home page. Two pre-built example profiles (Baseline, Vulnerable)
let users explore the demo without filling anything out, then they can edit
the form to make it their own.
"""

from __future__ import annotations

from datetime import time as dtime

import streamlit as st
from pydantic import ValidationError

from guardian.profile import (
    EmergencyContact,
    Home,
    HomeType,
    Location,
    Medical,
    Preferences,
    QuietHours,
    RiskNotifyLevel,
    UserProfile,
    Vehicle,
    VehicleClearance,
)


# ---------------------------------------------------------------------------
# Pre-built examples
# ---------------------------------------------------------------------------

def _example_baseline() -> UserProfile:
    return UserProfile(
        user_id="demo-baseline",
        name="Alex (baseline demo)",
        location=Location(
            address="100 Riverbend Dr, Baton Rouge, LA 70803",
            latitude=30.4133, longitude=-91.18,
            nws_zone_id="LAZ036", county_fips="22033",
        ),
        home=Home(type=HomeType.APARTMENT, floor_level=3),
        vehicle=Vehicle(owns_vehicle=True, clearance=VehicleClearance.HIGH,
                        four_wheel_drive=True),
        medical=Medical(),
        emergency_contacts=[EmergencyContact(
            name="Spouse", relationship="spouse", phone="+15555550100",
            notify_on=[RiskNotifyLevel.HIGH, RiskNotifyLevel.CRITICAL],
        )],
        preferences=Preferences(allow_smart_home_actions=True),
    )


def _example_vulnerable() -> UserProfile:
    return UserProfile(
        user_id="demo-vulnerable",
        name="Sam (vulnerable demo)",
        location=Location(
            address="42 Oak St, Baton Rouge, LA 70802",
            latitude=30.4515, longitude=-91.1871,
            nws_zone_id="LAZ036", county_fips="22033",
        ),
        home=Home(type=HomeType.APARTMENT, floor_level=1, flood_zone="AE"),
        vehicle=Vehicle(owns_vehicle=False, clearance=VehicleClearance.NONE),
        medical=Medical(mobility_limited=True, refrigerated_medication=True),
        emergency_contacts=[
            EmergencyContact(
                name="Sibling", relationship="sibling", phone="+15555550101",
                notify_on=[RiskNotifyLevel.MODERATE, RiskNotifyLevel.HIGH,
                           RiskNotifyLevel.CRITICAL],
            ),
            EmergencyContact(
                name="Parent", relationship="parent", phone="+15555550102",
                notify_on=[RiskNotifyLevel.CRITICAL],
            ),
        ],
        preferences=Preferences(
            allow_smart_home_actions=True,
            quiet_hours=QuietHours(start=dtime(22, 0), end=dtime(7, 0)),
        ),
    )


EXAMPLES: dict[str, UserProfile] = {
    "Baseline (low vulnerability)": _example_baseline(),
    "Vulnerable (ground floor, mobility-limited, no vehicle)": _example_vulnerable(),
}


# ---------------------------------------------------------------------------
# The form
# ---------------------------------------------------------------------------

def render_profile_form(prefilled: UserProfile | None = None) -> UserProfile | None:
    """Render the profile form. Returns a built UserProfile if the user
    submits valid data; None otherwise.
    """
    p = prefilled

    st.markdown("### Identity")
    name = st.text_input("Display name",
                         value=p.name if p else "Demo User")
    user_id = st.text_input("User ID (letters, numbers, _, -)",
                            value=p.user_id if p else "demo-user",
                            help="Used internally to tag log entries.")

    st.markdown("### Location")
    cols = st.columns([2, 1, 1])
    with cols[0]:
        address = st.text_input(
            "Home address",
            value=p.location.address if p else "100 Demo St, Baton Rouge, LA 70803",
        )
    with cols[1]:
        latitude = st.number_input(
            "Latitude", value=float(p.location.latitude) if p else 30.4133,
            min_value=-90.0, max_value=90.0, format="%.4f", step=0.0001,
        )
    with cols[2]:
        longitude = st.number_input(
            "Longitude", value=float(p.location.longitude) if p else -91.18,
            min_value=-180.0, max_value=180.0, format="%.4f", step=0.0001,
        )
    cols2 = st.columns(2)
    with cols2[0]:
        nws_zone_id = st.text_input(
            "NWS zone ID (optional, e.g. LAZ036)",
            value=p.location.nws_zone_id or "" if p else "LAZ036",
            help="Will be looked up from lat/lon if blank.",
        )
    with cols2[1]:
        county_fips = st.text_input(
            "County FIPS (optional, 5 digits)",
            value=p.location.county_fips or "" if p else "22033",
            help="Used by the ML threat classifier.",
        )

    st.markdown("### Home")
    cols = st.columns(3)
    with cols[0]:
        home_type = st.selectbox(
            "Home type",
            options=[t.value for t in HomeType],
            index=[t.value for t in HomeType].index(p.home.type.value) if p else 0,
        )
    with cols[1]:
        floor_level = st.number_input(
            "Floor level (1 = ground)",
            min_value=1, max_value=200,
            value=int(p.home.floor_level) if p else 1,
        )
    with cols[2]:
        elevated = st.checkbox(
            "Elevated home (pier-and-beam, stilts)",
            value=bool(p.home.elevated) if p else False,
        )
    cols2 = st.columns(3)
    with cols2[0]:
        flood_zone = st.text_input(
            "FEMA flood zone (optional)",
            value=p.home.flood_zone or "" if p else "",
        )
    with cols2[1]:
        has_generator = st.checkbox(
            "Backup generator", value=bool(p.home.has_generator) if p else False,
        )
    with cols2[2]:
        has_storm_shutters = st.checkbox(
            "Storm shutters", value=bool(p.home.has_storm_shutters) if p else False,
        )

    st.markdown("### Vehicle")
    cols = st.columns(3)
    with cols[0]:
        owns_vehicle = st.checkbox(
            "Owns a vehicle",
            value=bool(p.vehicle.owns_vehicle) if p else True,
        )
    with cols[1]:
        clearance_options = [c.value for c in VehicleClearance]
        clearance = st.selectbox(
            "Clearance",
            options=clearance_options,
            index=clearance_options.index(p.vehicle.clearance.value) if p else 1,
            disabled=not owns_vehicle,
        )
    with cols[2]:
        four_wheel_drive = st.checkbox(
            "Four-wheel drive",
            value=bool(p.vehicle.four_wheel_drive) if p else False,
            disabled=not owns_vehicle,
        )

    st.markdown("### Medical")
    cols = st.columns(3)
    with cols[0]:
        mobility_limited = st.checkbox(
            "Mobility limited",
            value=bool(p.medical.mobility_limited) if p else False,
        )
    with cols[1]:
        oxygen_dependent = st.checkbox(
            "Oxygen-dependent",
            value=bool(p.medical.oxygen_dependent) if p else False,
        )
    with cols[2]:
        refrigerated_medication = st.checkbox(
            "Refrigerated medication (e.g. insulin)",
            value=bool(p.medical.refrigerated_medication) if p else False,
        )
    chronic_raw = st.text_input(
        "Chronic conditions (comma-separated, optional)",
        value=", ".join(p.medical.chronic_conditions) if p else "",
    )

    st.markdown("### Emergency contacts")
    st.caption(
        "These contacts would receive SMS in the deployed agent. "
        "On this demo site, all SMS is dry-run only."
    )
    n_default = len(p.emergency_contacts) if p else 1
    n_contacts = st.number_input(
        "Number of contacts", min_value=0, max_value=5, value=n_default,
    )
    contacts: list[EmergencyContact] = []
    for i in range(int(n_contacts)):
        st.markdown(f"**Contact {i+1}**")
        existing = (p.emergency_contacts[i]
                    if p and i < len(p.emergency_contacts) else None)
        cols = st.columns([2, 2, 2])
        with cols[0]:
            c_name = st.text_input(
                "Name", key=f"c_name_{i}",
                value=existing.name if existing else f"Contact {i+1}",
            )
        with cols[1]:
            c_rel = st.text_input(
                "Relationship", key=f"c_rel_{i}",
                value=existing.relationship if existing else "friend",
            )
        with cols[2]:
            c_phone = st.text_input(
                "Phone (E.164, e.g. +15551234567)", key=f"c_phone_{i}",
                value=existing.phone if existing else "+15555550100",
            )
        notify_options = [r.value for r in RiskNotifyLevel]
        default_notify = (
            [r.value for r in existing.notify_on] if existing
            else ["high", "critical"]
        )
        c_notify = st.multiselect(
            "Notify on which risk levels?", options=notify_options,
            default=default_notify, key=f"c_notify_{i}",
        )
        try:
            contacts.append(EmergencyContact(
                name=c_name, relationship=c_rel, phone=c_phone,
                notify_on=[RiskNotifyLevel(v) for v in c_notify] or [
                    RiskNotifyLevel.HIGH, RiskNotifyLevel.CRITICAL,
                ],
            ))
        except ValidationError as e:
            st.error(f"Contact {i+1} is invalid: {_first_pydantic_error(e)}")
            return None

    st.markdown("### Preferences")
    cols = st.columns(2)
    with cols[0]:
        smart_home = st.checkbox(
            "Allow smart-home actions (lights, alarm, thermostat)",
            value=bool(p.preferences.allow_smart_home_actions) if p else True,
        )
    with cols[1]:
        use_quiet_hours = st.checkbox(
            "Use quiet hours",
            value=p.preferences.quiet_hours is not None if p else False,
        )
    quiet_hours: QuietHours | None = None
    if use_quiet_hours:
        cols2 = st.columns(2)
        with cols2[0]:
            qh_start_default = (
                p.preferences.quiet_hours.start if p and p.preferences.quiet_hours
                else dtime(22, 0)
            )
            qh_start = st.time_input("Quiet hours start", value=qh_start_default)
        with cols2[1]:
            qh_end_default = (
                p.preferences.quiet_hours.end if p and p.preferences.quiet_hours
                else dtime(7, 0)
            )
            qh_end = st.time_input("Quiet hours end", value=qh_end_default)
        quiet_hours = QuietHours(start=qh_start, end=qh_end)

    # ---- Validate and build ----
    try:
        profile = UserProfile(
            user_id=user_id,
            name=name,
            location=Location(
                address=address,
                latitude=latitude,
                longitude=longitude,
                nws_zone_id=nws_zone_id or None,
                county_fips=county_fips or None,
            ),
            home=Home(
                type=HomeType(home_type),
                floor_level=int(floor_level),
                elevated=elevated,
                flood_zone=flood_zone or None,
                has_generator=has_generator,
                has_storm_shutters=has_storm_shutters,
            ),
            vehicle=Vehicle(
                owns_vehicle=owns_vehicle,
                clearance=VehicleClearance(clearance),
                four_wheel_drive=four_wheel_drive,
            ),
            medical=Medical(
                mobility_limited=mobility_limited,
                oxygen_dependent=oxygen_dependent,
                refrigerated_medication=refrigerated_medication,
                chronic_conditions=[
                    c.strip() for c in chronic_raw.split(",") if c.strip()
                ],
            ),
            emergency_contacts=contacts,
            preferences=Preferences(
                allow_smart_home_actions=smart_home,
                quiet_hours=quiet_hours,
            ),
        )
        return profile
    except ValidationError as e:
        st.error(f"Profile invalid: {_first_pydantic_error(e)}")
        return None


def _first_pydantic_error(e: ValidationError) -> str:
    errs = e.errors()
    if not errs:
        return str(e)
    first = errs[0]
    loc = ".".join(str(x) for x in first.get("loc", []))
    msg = first.get("msg", "validation error")
    return f"{loc}: {msg}"
