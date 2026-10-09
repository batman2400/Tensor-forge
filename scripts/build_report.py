"""Write the five-page Phase 2 report from the recorded evidence in this file.

The numbers are the measurements taken on 9 October 2026. They are not estimated.
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "TensorForge_Phase2_Final_Report.pdf"

NAVY = (18, 42, 74)
RULE = (180, 190, 200)


class Report(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(90, 90, 90)
        self.cell(
            0, 6, f"Team Fade  |  TensorForge 2.0 Phase 2  |  {self.page_no()} / {{nb}}", align="C"
        )


def heading(pdf: Report, text: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(0, 6, text, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*RULE)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)


def body(pdf: Report, text: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(0, 4.3, text, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.4)


def table(pdf: Report, rows: list[list[str]], widths: list[int]) -> None:
    pdf.set_font("Helvetica", "", 8)
    for index, row in enumerate(rows):
        pdf.set_x(pdf.l_margin)
        if index == 0:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(232, 237, 243)
        else:
            pdf.set_font("Helvetica", "", 8)
            pdf.set_fill_color(255, 255, 255)
        for value, width in zip(row, widths, strict=True):
            pdf.cell(width, 5.2, value, border=0, fill=True)
        pdf.ln(5.2)
    pdf.ln(1.5)


def main() -> None:
    pdf = Report(format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(14, 14, 14)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(0, 7, "TensorForge 2.0 / Phase 2 MVP", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 5, "Support ticket classification and routing", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0,
        5,
        "Team Fade  |  Final technical report  |  9 October 2026",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2)

    heading(pdf, "1. Solution and design decisions")
    body(
        pdf,
        "TensorForge classifies RideEat support tickets, a fictional ride-hailing and food-delivery platform. "
        "For each ticket it assigns one of 11 primary categories, maps that category to the organizer-defined team, "
        "returns an optional secondary category, predicts urgency, and returns a confidence with a human-review flag "
        "when confidence is below 0.5. It accepts email, chat, and call transcripts in English, Sinhala, Tamil, "
        "and transliterated or mixed text. No generative LLM or external classification API is called at inference.",
    )
    body(
        pdf,
        "The deployed model fuses a self-trained word and character TF-IDF classifier with a fine-tuned "
        "multilingual encoder. The development-validation macro-F1 is 0.8022, compared with 0.6558 for the "
        "selected classical baseline. The model version is v1.0.0-38ecb9bd. This 0.8022 figure is development "
        "validation, not an untouched holdout: fusion weights, temperatures, and thresholds were selected using "
        "out-of-fold predictions and labels across all 4,800 tickets, including the validation labels.",
    )
    table(
        pdf,
        [
            ["Decision", "Reason"],
            [
                "Classical + encoder fusion",
                "Character n-grams cover spelling; the encoder adds multilingual meaning.",
            ],
            [
                "Macro-F1 for selection",
                "Classes range from about 3% to 18%, so accuracy would hide rare classes.",
            ],
            [
                "Contract middleware",
                "Auth runs before parsing. Errors stay JSON with the required status codes.",
            ],
            [
                "CPU-only offline image",
                "Int8 ONNX and local files. The image serves with networking disabled.",
            ],
            [
                "One worker + SQLite jobs",
                "Large jobs poll from durable state and are marked interrupted after restart.",
            ],
        ],
        [52, 130],
    )
    body(
        pdf,
        "Live service: https://tensorforge-fade.southindia.cloudapp.azure.com  |  Demo: /demo/  |  "
        "Repository: github.com/batman2400/Tensor-forge  |  Release: v1.0.1",
    )

    heading(pdf, "2. Data, training, and scores")
    body(
        pdf,
        "The organizer set has 4,000 training tickets and 800 validation tickets. Across all 4,800, secondary "
        "labels and urgent tickets are each about 10%. Languages are English 35%, Singlish 25%, Sinhala 20%, "
        "Tamil 15%, mixed 2.5%, and Tanglish 2.5%. Channels are chat 45%, email 30%, and call transcript 25%. "
        "ticket_id and the language label are not model features. Text is NFC-normalized. The API allows 10,000 characters.",
    )
    body(
        pdf,
        "The classical branch is svc_word_char (word and character TF-IDF, linear SVM). Logistic regression, SGD, "
        "naive Bayes, and LightGBM were compared on the same split. The encoder is intfloat/multilingual-e5-small "
        "(MIT), fine-tuned with task heads and exported to int8 ONNX. The serving encoder was refit on all 4,800 "
        "tickets for four epochs. Fusion gives the encoder 0.75 of the category and secondary vote and 0.50 of the urgency vote.",
    )
    table(
        pdf,
        [
            ["Configuration", "Accuracy", "Macro-F1", "Urgent F1", "Secondary exact"],
            ["Word TF-IDF logistic regression", "0.4600", "0.5014", "0.8252", "0.9525"],
            ["Word + char logistic regression", "0.5787", "0.6037", "0.8725", "0.9513"],
            ["Word + char SVC baseline", "0.6338", "0.6558", "0.8295", "0.9550"],
            ["SVC + encoder fusion", "0.7975", "0.8022", "0.9080", "0.9637"],
        ],
        [68, 28, 28, 28, 30],
    )
    body(
        pdf,
        "Recorded fusion ECE is 0.0589. Sinhala is the weakest language slice (macro-F1 0.624 on 160 tickets). "
        "Mixed and Tanglish each have only 20 tickets. A score of the final all-data encoder on the same 800 rows "
        "is in-sample; an API check of those rows scored about 0.994 macro-F1 and is not validation accuracy. "
        "Pooled random-fold scores are higher because tickets share opening templates. Leave-one-language-out "
        "retraining was not run. An untouched organizer holdout remains the decisive generalization test.",
    )

    pdf.add_page()
    heading(pdf, "3. Container evidence (9 October 2026)")
    body(
        pdf,
        "Image sha256:45131d2759b9a58d5d40529f5c740f2ee7c5ceb7464eef1b0a80d793dc31a3db (983 MB), built from "
        "commit 52e612b5fccbe228bfc2bb3520df1427726845a9 with Git LFS weights hydrated. The image label records "
        "that commit and version v1.0.1. A scan of the image filesystem found no tf2_ key and no API_KEY. "
        "The base Python image sets GPG_KEY to the public CPython release signing key; that is not a project secret. "
        "The service key is supplied only at runtime.",
    )
    table(
        pdf,
        [
            ["Check under 2 CPU / 4 GB, swap capped at 4 GB", "Result"],
            ["Startup to HTTP 200 /health", "4.3 s (limit 120 s)"],
            ["Startup with --network none", "3.3 s, health 200"],
            ["Contract, auth, 100-ticket batch, jobs", "256/256 checks passed"],
            ["2,000-ticket job / poll p95", "33 s / 0.02 s"],
            ["5,000-ticket job / poll p95", "80 s / 0.02 s; submit under 5 s"],
            ["Peak resident memory during those jobs", "572 MiB"],
            [
                "Restart during a 1,500-ticket job",
                "failed / interrupted; health in 3.1 s",
            ],
        ],
        [100, 82],
    )
    body(
        pdf,
        "GitHub Actions run 37921671034 on commit 52e612b succeeded for both the hygiene job and the Linux test "
        "job (LFS checkout, non-slow pytest). The in-sample 0.994 figure above came from this same container run.",
    )

    heading(pdf, "4. Hosted service")
    body(
        pdf,
        "The tested image was loaded on the South India VM and the container was recreated with restart policy "
        "unless-stopped, 2 CPUs, and a 3 GB memory cap. The VM has 3.8 GB of RAM, so the live cap is 3 GB rather "
        "than the 4 GB proof above. Job storage stayed on the Docker volume tf_data. The previous image "
        "sha256:e18b0c716e0240c4dc86f9a04703380e9a82cd2a604856e4d30d3e21da88a11c remains on the VM as the rollback "
        "image. docker image inspect on the VM reports the same digest as the local build. Public HTTPS /health "
        "returned 200 with model_version v1.0.0-38ecb9bd.",
    )
    body(
        pdf,
        "Earlier laptop failures were the uplink, not the application. From this network a 30 MB upload took about "
        "81 s, so a 60 s client write timeout fired before the JSON 401 could be read. On the VM, through Caddy, "
        "the deployed image returned JSON 401 for a 30 MB unauthenticated upload in 0.15 s and JSON 413 for an "
        "authenticated 6 MB batch. Direct container checks on the VM were the same shape, in under 0.03 s.",
    )
    body(
        pdf,
        "Latency was measured again on the VM, through public HTTPS, after this image was deployed. "
        "Single-prediction p95 was 0.035 s (40 requests). A 100-ticket batch p95 was 1.53 s (5 requests). "
        "The 2,000-ticket job was accepted in 0.017 s and finished in 48.5 s. The 5,000-ticket job was accepted "
        "in 0.031 s and finished in 119.2 s. Poll p95 during that job was 0.012 s, and /health p95 was 0.010 s. "
        "All of those are under the one-second poll target and the five-second submit target. CPU credits "
        "remaining before the run were about 921.",
    )
    body(
        pdf,
        "Auto-shutdown: no DevTestLab shutdown schedule exists on the resource group. SSH is allowed from one "
        "operator address only. Ports 80 and 443 are open. Caddy terminates TLS and proxies to 127.0.0.1:8000 "
        "with a 64 MB body limit, above the application's limits, and a 120 s response header timeout. An Azure "
        "Monitor alert on VmAvailabilityMetric notifies the subscription operator.",
    )

    heading(pdf, "5. Release identity and what is still open")
    body(
        pdf,
        "Tag v1.0.0 still points at f164ba49b01626e8926335e46639fd2655fc076e, which is before the artifact-byte fix. "
        "It was not moved. This submission is release v1.0.1. The image digest to submit is "
        "sha256:45131d2759b9a58d5d40529f5c740f2ee7c5ceb7464eef1b0a80d793dc31a3db. An older plan entry, "
        "sha256:5d64978226bc891c3d7b09abf12ec19eb25d0c16aa8e523b6094d284b0639920, is not the running image. "
        "The fresh-clone hash mismatch described in an earlier draft of this report was fixed before commit "
        "cc6da4f; current artifact hashes match the manifest, and the image build runs Engine.load() before it succeeds.",
    )
    body(
        pdf,
        "Authenticated hosted checks in this session succeeded, including the upload cases above. The video is not "
        "in this report: it had not been recorded at publication time. The script in docs/video_script.md uses "
        "synthetic tickets in docs/demo_tickets.csv, tells the presenter to keep the API key off camera, and "
        "calls 0.802 development validation rather than an untouched holdout. The presenter still has to record "
        "about 15 minutes, upload it unlisted, and confirm playback while logged out of YouTube.",
    )
    body(
        pdf,
        "Hosting-credit balance was not visible to the Azure login used for this deploy, so remaining credit is "
        "not stated as a number here. The VM was running, CPU credits were nearly full, and disk use was 6.2 GB of 29 GB.",
    )
    pdf.output(OUT)
    print(f"wrote {OUT} pages={pdf.pages_count}")


if __name__ == "__main__":
    main()
