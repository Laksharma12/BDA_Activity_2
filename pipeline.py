import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import QuantileTransformer


ROOT = Path(__file__).resolve().parents[1]
EPISODES = ROOT / "data" / "raw" / "episodes"
PROCESSED = ROOT / "data" / "processed"
CHARTS = ROOT / "outputs" / "charts"
MAPREDUCE = ROOT / "outputs" / "mapreduce"
MODEL = ROOT / "outputs" / "model"
REPORT = ROOT / "report"


def flatten_actions(action):
    for group, values in action.items():
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, list) and value:
                yield group, str(value[0]), ":".join(map(str, value))
            elif isinstance(value, str):
                yield group, value, value


def safe_number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else np.nan


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)


def main():
    for directory in (PROCESSED, CHARTS, MAPREDUCE, MODEL, REPORT):
        directory.mkdir(parents=True, exist_ok=True)

    files = sorted(EPISODES.glob("*.json"))
    episode_rows, player_rows, turn_rows = [], [], []
    action_counter = Counter()
    price_rows = []
    malformed = []

    for path in files:
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
            steps = episode.get("steps", [])
            episode_id = episode.get("info", {}).get("EpisodeId", episode.get("id", path.stem))
            rewards = episode.get("rewards", [])
            statuses = episode.get("statuses", [])
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "file_name": path.name,
                    "steps": len(steps),
                    "players": len(steps[0]) if steps and isinstance(steps[0], list) else 0,
                    "status_0": statuses[0] if statuses else "",
                    "status_1": statuses[1] if len(statuses) > 1 else "",
                    "reward_0": rewards[0] if rewards else np.nan,
                    "reward_1": rewards[1] if len(rewards) > 1 else np.nan,
                    "schema_version": episode.get("schema_version"),
                }
            )
            player_stats = defaultdict(lambda: {"actions": 0, "action_types": set(), "money": [], "rewards": []})
            for turn_index, turn in enumerate(steps):
                if not isinstance(turn, list):
                    continue
                for obs in turn:
                    if not isinstance(obs, dict):
                        continue
                    observation = obs.get("observation", {})
                    player = observation.get("player")
                    if player is None:
                        continue
                    farm_list = observation.get("farms", [])
                    farm = farm_list[player] if isinstance(farm_list, list) and player < len(farm_list) else {}
                    actions = list(flatten_actions(obs.get("action", {})))
                    for group, action_type, action_value in actions:
                        action_counter[(group, action_type)] += 1
                        player_stats[player]["actions"] += 1
                        player_stats[player]["action_types"].add(action_type)
                    money = safe_number(farm.get("money"))
                    reward = safe_number(obs.get("reward"))
                    player_stats[player]["money"].append(money)
                    player_stats[player]["rewards"].append(reward)
                    turn_rows.append(
                        {
                            "episode_id": episode_id,
                            "player": player,
                            "turn": observation.get("step", turn_index),
                            "day": observation.get("day"),
                            "hour": observation.get("hour"),
                            "money": money,
                            "reward": reward,
                            "status": obs.get("status", ""),
                            "action_count": len(actions),
                        }
                    )
                    prices = observation.get("market", {}).get("prices", {})
                    for product, price in prices.items():
                        price_rows.append(
                            {
                                "episode_id": episode_id,
                                "player": player,
                                "turn": observation.get("step", turn_index),
                                "product": product,
                                "price": safe_number(price),
                            }
                        )
            for player, stats in player_stats.items():
                money = [x for x in stats["money"] if not math.isnan(x)]
                rewards_for_player = [x for x in stats["rewards"] if not math.isnan(x)]
                player_rows.append(
                    {
                        "episode_id": episode_id,
                        "player": player,
                        "final_money": money[-1] if money else np.nan,
                        "initial_money": money[0] if money else np.nan,
                        "mean_money": np.mean(money) if money else np.nan,
                        "total_actions": stats["actions"],
                        "unique_action_types": len(stats["action_types"]),
                        "mean_step_reward": np.mean(rewards_for_player) if rewards_for_player else np.nan,
                        "terminal_reward": rewards[player] if player < len(rewards) else np.nan,
                        "status": statuses[player] if player < len(statuses) else "",
                    }
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            malformed.append({"file_name": path.name, "error": str(exc)})

    episodes = pd.DataFrame(episode_rows)
    players = pd.DataFrame(player_rows)
    turns = pd.DataFrame(turn_rows)
    prices = pd.DataFrame(price_rows)
    actions = pd.DataFrame(
        [{"action_group": group, "action_type": action, "count": count} for (group, action), count in action_counter.items()]
    ).sort_values("count", ascending=False)

    episodes.to_csv(PROCESSED / "episode_summary.csv", index=False)
    players.to_csv(PROCESSED / "player_performance.csv", index=False)
    turns.to_csv(PROCESSED / "turn_level_data.csv", index=False)
    actions.to_csv(PROCESSED / "farming_actions.csv", index=False)
    prices.to_csv(PROCESSED / "market_analysis.csv", index=False)
    pd.DataFrame(malformed).to_csv(PROCESSED / "malformed_files.csv", index=False)

    # Explicit Mapper -> Shuffle/Sort -> Reducer demonstration.
    mapped = [(row.player, row.terminal_reward) for row in players.itertuples() if pd.notna(row.terminal_reward)]
    shuffled = defaultdict(list)
    for key, value in mapped:
        shuffled[key].append(float(value))
    reduced = [{"player": key, "record_count": len(values), "total_terminal_reward": sum(values),
                "average_terminal_reward": np.mean(values)} for key, values in sorted(shuffled.items())]
    pd.DataFrame(reduced).to_csv(MAPREDUCE / "player_reward_reducer_output.csv", index=False)
    with (MAPREDUCE / "mapreduce_trace.txt").open("w", encoding="utf-8") as trace:
        trace.write("Mapper sample: " + repr(mapped[:10]) + "\n")
        trace.write("Shuffle/sort keys: " + repr(sorted(shuffled)) + "\n")
        trace.write("Reducer output: " + repr(reduced) + "\n")

    # Cluster one row per player-episode, using only observed, post-step summary features.
    # Keep one representative feature from each signal family.  Initial money is
    # constant in this dataset, while terminal_reward duplicates final_money.
    feature_columns = ["final_money", "mean_money", "total_actions", "unique_action_types"]
    model_data = players[["episode_id", "player"] + feature_columns].dropna().copy()
    if len(model_data) >= 6:
        # Money and action counts are strongly right-skewed.  Log compression
        # prevents their largest values from dominating distance calculations;
        # quantile normalization then puts each valid feature on a comparable
        # approximately normal scale.
        transformed = np.log1p(model_data[feature_columns].clip(lower=0))
        normalizer = QuantileTransformer(
            n_quantiles=min(50, len(model_data)),
            output_distribution="normal",
            random_state=42,
        )
        x = normalizer.fit_transform(transformed)
        scores = []
        for k in range(2, min(10, len(model_data) - 1) + 1):
            labels = KMeans(n_clusters=k, random_state=42, n_init=50).fit_predict(x)
            scores.append({"k": k, "silhouette_score": silhouette_score(x, labels)})
        score_df = pd.DataFrame(scores)
        score_df.to_csv(MODEL / "silhouette_scores.csv", index=False)
        best_k = int(score_df.loc[score_df["silhouette_score"].idxmax(), "k"])
        model = KMeans(n_clusters=best_k, random_state=42, n_init=50)
        model_data["cluster"] = model.fit_predict(x)
        model_data.to_csv(MODEL / "player_episode_clusters.csv", index=False)
    else:
        score_df = pd.DataFrame()
        model_data = pd.DataFrame()

    plt.style.use("seaborn-v0_8-whitegrid")
    chart_paths = []
    def savefig(name):
        path = CHARTS / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        chart_paths.append(path)

    if not players.empty:
        players.groupby("player")["terminal_reward"].mean().plot(kind="bar", title="Average Terminal Reward by Player", ylabel="Reward", xlabel="Player")
        savefig("player_performance.png")
        players["terminal_reward"].dropna().plot(kind="hist", bins=30, title="Terminal Reward Distribution", xlabel="Reward")
        savefig("reward_distribution.png")
    if not actions.empty:
        actions.head(15).sort_values("count").plot(kind="barh", x="action_type", y="count", legend=False, title="Most Frequent Action Types", xlabel="Count")
        savefig("farming_action_frequency.png")
    if not turns.empty:
        turns.groupby("day")["action_count"].sum().plot(title="Daily Farming Activity", xlabel="Day", ylabel="Action count")
        savefig("daily_activity.png")
    if not prices.empty:
        prices.groupby("product")["price"].mean().sort_values().plot(kind="bar", title="Average Market Price by Product", xlabel="Product", ylabel="Price")
        savefig("market_price_comparison.png")
    if not players.empty:
        corr = players[feature_columns].corr(numeric_only=True)
        plt.figure(figsize=(8, 6))
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Correlation")
        plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr)), corr.columns)
        plt.title("Player Feature Correlation Heatmap")
        savefig("correlation_heatmap.png")
    if not score_df.empty:
        score_df.plot(x="k", y="silhouette_score", marker="o", legend=False, title="K-Means Silhouette Scores", xlabel="Number of clusters", ylabel="Silhouette score")
        savefig("kmeans_silhouette.png")

    # Self-contained PDF report with executed metrics and embedded visuals.
    with PdfPages(REPORT / "CA3_Analytical_Report.pdf") as pdf:
        pages = [
            ("Kaggriculture Analytics Report", [
                f"Generated from {len(files)} JSON replay files.",
                f"Valid episodes: {len(episodes)} | Malformed files: {len(malformed)}",
                f"Turn-player records: {len(turns):,} | Player-episode records: {len(players):,}",
                "Results are descriptive and do not establish causation.",
            ]),
            ("Data quality and schema", [
                "Root fields observed: configuration, description, id, info, module_version, name, rewards, schema_version, specification, statuses, steps, title, version.",
                f"Step counts: min={episodes.steps.min() if not episodes.empty else 0}, max={episodes.steps.max() if not episodes.empty else 0}.",
                "Each inspected step contains observations, actions, rewards and status; observations contain day, hour, farms, market, player and private state.",
                "Malformed JSON files are listed in data/processed/malformed_files.csv.",
            ]),
            ("Analytics and MapReduce", [
                "Mapper key: player; value: terminal reward.",
                "Shuffle/sort groups values by player; reducer computes count, total and average terminal reward.",
                f"Distinct action types: {actions.action_type.nunique() if not actions.empty else 0}.",
                "Full mapper, shuffle and reducer samples are saved in outputs/mapreduce/mapreduce_trace.txt.",
            ]),
            ("Machine learning", [
                "K-Means clusters one row per player-episode after standardization.",
                f"Candidate cluster scores: {score_df.to_dict('records') if not score_df.empty else 'not enough complete records'}.",
                "Cluster names must be assigned from the generated cluster profiles rather than assumed in advance.",
            ]),
        ]
        for title, lines in pages:
            fig = plt.figure(figsize=(11.7, 8.3))
            fig.text(0.06, 0.9, title, fontsize=22, weight="bold")
            fig.text(0.07, 0.82, "\n\n".join(lines), fontsize=13, va="top", wrap=True)
            pdf.savefig(fig)
            plt.close(fig)
        for chart in chart_paths:
            image = plt.imread(chart)
            fig = plt.figure(figsize=(11.7, 8.3))
            fig.figimage(image, xo=80, yo=80, resize=True)
            fig.text(0.06, 0.94, chart.stem.replace("_", " ").title(), fontsize=18, weight="bold")
            pdf.savefig(fig)
            plt.close(fig)

    manifest = {
        "json_files_seen": len(files),
        "valid_episodes": len(episodes),
        "malformed_files": len(malformed),
        "turn_player_rows": len(turns),
        "player_episode_rows": len(players),
        "charts": [str(p.relative_to(ROOT)) for p in chart_paths],
        "report": str((REPORT / "CA3_Analytical_Report.pdf").relative_to(ROOT)),
    }
    (ROOT / "outputs" / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
