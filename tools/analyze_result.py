#!/usr/bin/env python3
"""
Analyze closed-loop evaluation results from merged.json without requiring CARLA.
This is a lightweight alternative to ability_benchmark.py.

Usage:
    python tools/analyze_results.py -r <path_to_merged.json>
"""

import json
import argparse
from collections import defaultdict

Ability = {
    "Overtaking": ['Accident', 'AccidentTwoWays', 'ConstructionObstacle', 'ConstructionObstacleTwoWays',
                    'HazardAtSideLaneTwoWays', 'HazardAtSideLane', 'ParkedObstacleTwoWays', 'ParkedObstacle',
                    'VehicleOpensDoorTwoWays'],
    "Merging": ['CrossingBicycleFlow', 'EnterActorFlow', 'HighwayExit', 'InterurbanActorFlow', 'HighwayCutIn',
                'InterurbanAdvancedActorFlow', 'MergerIntoSlowTrafficV2', 'MergerIntoSlowTraffic',
                'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn',
                'NonSignalizedJunctionLeftTurnEnterFlow', 'ParkingExit', 'SequentialLaneChange',
                'SignalizedJunctionLeftTurn', 'SignalizedJunctionRightTurn', 'SignalizedJunctionLeftTurnEnterFlow'],
    "Emergency_Brake": ['BlockedIntersection', 'DynamicObjectCrossing', 'HardBreakRoute',
                         'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'ParkingCutIn',
                         'PedestrianCrossing', 'ParkingCrossingPedestrian', 'StaticCutIn', 'VehicleTurningRoute',
                         'VehicleTurningRoutePedestrian', 'ControlLoss'],
    "Give_Way": ['InvadingTurn', 'YieldToEmergencyVehicle'],
    "Traffic_Signs": ['BlockedIntersection', 'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight',
                      'PedestrianCrossing', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian',
                      'EnterActorFlow', 'CrossingBicycleFlow', 'NonSignalizedJunctionLeftTurn',
                      'NonSignalizedJunctionRightTurn', 'NonSignalizedJunctionLeftTurnEnterFlow',
                      'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight',
                      'PedestrianCrossing', 'SignalizedJunctionLeftTurn', 'SignalizedJunctionRightTurn',
                      'SignalizedJunctionLeftTurnEnterFlow', 'T_Junction', 'VanillaNonSignalizedTurn',
                      'VanillaSignalizedTurnEncounterGreenLight', 'VanillaSignalizedTurnEncounterRedLight',
                      'VanillaNonSignalizedTurnEncounterStopsign', 'VehicleTurningRoute',
                      'VehicleTurningRoutePedestrian']
}


def has_infractions(record):
    """Check if a record has any infractions (excluding min_speed_infractions)."""
    for infraction, value in record['infractions'].items():
        if infraction == "min_speed_infractions":
            continue
        if len(value) > 0:
            return True
    return False


def main(args):
    with open(args.result_file, 'r') as f:
        data = json.load(f)

    records = data["_checkpoint"]["records"]
    total = len(records)

    # --- Overall Statistics ---
    status_counts = defaultdict(int)
    completed = []
    crashed = []
    for r in records:
        status_counts[r["status"]] += 1
        if r["status"] in ("Completed", "Perfect"):
            completed.append(r)
        else:
            crashed.append(r)

    print("=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"  Total routes evaluated:  {total}")
    print(f"  Driving Score:           {data.get('driving score', 'N/A'):.4f}")
    print(f"  Success Rate:            {data.get('success rate', 'N/A'):.4f}")
    print()
    print("  Status breakdown:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"    {status}: {count} ({count/total*100:.1f}%)")

    # --- Crash Analysis ---
    crash_durations = [r["meta"]["duration_system"] for r in crashed]
    early_crash = [r for r in crashed if r["meta"]["duration_system"] <= 3.1]
    print()
    print(f"  Crashed routes: {len(crashed)}")
    print(f"    Early crash (≤3s): {len(early_crash)} ({len(early_crash)/max(len(crashed),1)*100:.1f}%)")
    if crash_durations:
        print(f"    Duration range: {min(crash_durations):.2f}s - {max(crash_durations):.2f}s")

    # --- Completed Routes ---
    print()
    print("=" * 70)
    print("COMPLETED ROUTES")
    print("=" * 70)
    if completed:
        for r in completed:
            infraction_flag = "✗ has infractions" if has_infractions(r) else "✓ clean"
            print(f"  [{r['scores']['score_composed']:6.2f}] {r['save_name']}")
            print(f"          {r['scenario_name']} | {r['town_name']} | {r['meta']['duration_system']:.1f}s | {infraction_flag}")
            print(f"          route: {r['scores']['score_route']:.1f}% | infractions: {r['num_infractions']}")
    else:
        print("  (none)")

    # --- Per-Scenario Statistics ---
    scenario_stats = defaultdict(lambda: {"total": 0, "completed": 0, "success": 0, "scores": []})
    for r in records:
        s = r["scenario_name"]
        # Remove trailing _1, _2 etc for grouping
        scenario_stats[s]["total"] += 1
        if r["status"] in ("Completed", "Perfect"):
            scenario_stats[s]["completed"] += 1
            scenario_stats[s]["scores"].append(r["scores"]["score_composed"])
            if not has_infractions(r):
                scenario_stats[s]["success"] += 1

    print()
    print("=" * 70)
    print("PER-SCENARIO STATISTICS")
    print("=" * 70)
    for scenario, stats in sorted(scenario_stats.items(), key=lambda x: -x[1]["total"]):
        comp_rate = stats["completed"] / stats["total"] * 100
        succ_rate = stats["success"] / stats["total"] * 100
        avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
        print(f"  {scenario}:")
        print(f"    total={stats['total']}, completed={stats['completed']} ({comp_rate:.0f}%), "
              f"success(no infraction)={stats['success']} ({succ_rate:.0f}%), avg_score={avg_score:.2f}")

    # --- Per-Town Statistics ---
    town_stats = defaultdict(lambda: {"total": 0, "completed": 0, "crashed": 0})
    for r in records:
        t = r["town_name"]
        town_stats[t]["total"] += 1
        if r["status"] in ("Completed", "Perfect"):
            town_stats[t]["completed"] += 1
        else:
            town_stats[t]["crashed"] += 1

    print()
    print("=" * 70)
    print("PER-TOWN STATISTICS")
    print("=" * 70)
    for town, stats in sorted(town_stats.items()):
        comp_rate = stats["completed"] / stats["total"] * 100
        print(f"  {town}: total={stats['total']}, completed={stats['completed']} ({comp_rate:.0f}%), "
              f"crashed={stats['crashed']}")

    # --- Ability Benchmark (simplified, without Traffic_Signs junction check) ---
    print()
    print("=" * 70)
    print("ABILITY BENCHMARK (simplified, without CARLA junction check)")
    print("=" * 70)
    print("  Note: Traffic_Signs requires CARLA for accurate junction-based evaluation.")
    print("        Here we report completion-based success only (no infraction + completed).")
    print()

    ability_stats = {}
    for ability, scenarios in Ability.items():
        success = 0
        total_count = 0
        for r in records:
            if r["scenario_name"] in scenarios:
                total_count += 1
                if r["status"] in ("Completed", "Perfect") and not has_infractions(r):
                    success += 1
        if total_count > 0:
            rate = success / total_count
            ability_stats[ability] = rate
            print(f"  {ability}: {success}/{total_count} = {rate:.4f}")
        else:
            ability_stats[ability] = 0.0
            print(f"  {ability}: 0/0 = N/A")

    if ability_stats:
        mean_ability = sum(ability_stats.values()) / len(ability_stats)
        print(f"\n  Mean Ability Score: {mean_ability:.4f}")

    # --- Save output ---
    output_file = args.result_file.replace('.json', '_analysis.json')
    output = {
        "overall": {
            "total": total,
            "driving_score": data.get("driving score"),
            "success_rate": data.get("success rate"),
            "status_counts": dict(status_counts),
        },
        "completed_routes": [
            {
                "save_name": r["save_name"],
                "scenario": r["scenario_name"],
                "town": r["town_name"],
                "score_composed": r["scores"]["score_composed"],
                "score_route": r["scores"]["score_route"],
                "duration": r["meta"]["duration_system"],
                "has_infractions": has_infractions(r),
                "num_infractions": r["num_infractions"],
            }
            for r in completed
        ],
        "ability_simplified": ability_stats,
        "per_scenario": {
            s: {"total": v["total"], "completed": v["completed"], "success": v["success"]}
            for s, v in scenario_stats.items()
        },
        "per_town": dict(town_stats),
    }
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"\n  Results saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze closed-loop evaluation results (no CARLA required)')
    parser.add_argument('-r', '--result_file', required=True, help='Path to merged.json result file')
    args = parser.parse_args()
    main(args)
