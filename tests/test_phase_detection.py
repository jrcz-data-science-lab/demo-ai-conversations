import unittest

from app.manager.phase_detection import (
    DEFAULT_PHASE_CONFIG,
    analyze_conversation_phases,
    detect_gordon_patterns,
)


class PhaseDetectionTests(unittest.TestCase):
    def test_phase1_strong_phase3_weak(self):
        transcript = [
            {
                "speaker": "student",
                "text": "Goedemorgen, ik ben Anna, student verpleegkunde. Hoe gaat het met u vandaag?",
            },
            {
                "speaker": "student",
                "text": "We hebben ongeveer twintig minuten; het doel van dit gesprek is uw klachten bespreken. Wat verwacht u?",
            },
            {"speaker": "patient", "text": "Het gaat wel."},
            {"speaker": "student", "text": "Vertelt u gerust verder."},
            {"speaker": "patient", "text": "Ik heb wat vragen."},
        ]
        analysis = analyze_conversation_phases(transcript, DEFAULT_PHASE_CONFIG)
        phase1_score = analysis["phases"]["phase1"]["score_total"]
        phase3_score = analysis["phases"]["phase3"]["score_total"]
        self.assertGreaterEqual(phase1_score, 4)
        self.assertEqual(phase3_score, 0)

    def test_phase2_open_questions_and_summaries(self):
        transcript = [
            {"speaker": "student", "text": "Hallo, ik ben Tom, student verpleegkunde."},
            {"speaker": "student", "text": "Hoe gaat het met u vandaag?"},
            {"speaker": "patient", "text": "Ik ben wat benauwd."},
            {"speaker": "student", "text": "Wat maakt dat u zich benauwd voelt?"},
            {"speaker": "patient", "text": "Vooral 's nachts."},
            {"speaker": "student", "text": "Als ik u goed begrijp is het vooral 's nachts, klopt dat?"},
            {"speaker": "student", "text": "Even samenvatten: u bent benauwd en slaapt slecht, klopt dat?"},
            {"speaker": "patient", "text": "Ja, dat klopt."},
            {"speaker": "student", "text": "Kunt u een voorbeeld geven van een nacht?"},  # open question
            {"speaker": "student", "text": "Samenvattend: benauwdheid en slecht slapen."},
            {"speaker": "student", "text": "Heeft u nog vragen? Dank u wel."},
        ]
        analysis = analyze_conversation_phases(transcript, DEFAULT_PHASE_CONFIG)
        phase2_items = analysis["phases"]["phase2"]["items"]
        self.assertEqual(phase2_items["open_questions"]["score"], 2)
        self.assertGreaterEqual(phase2_items["interim_summaries"]["score"], 1)
        self.assertGreaterEqual(phase2_items["understanding_checks"]["score"], 1)

    def test_active_listening_cap(self):
        transcript = []
        for _ in range(10):
            transcript.append({"speaker": "student", "text": "ja"})
            transcript.append({"speaker": "patient", "text": "antwoord"})
        analysis = analyze_conversation_phases(transcript, DEFAULT_PHASE_CONFIG)
        # Count should be capped to config value
        self.assertEqual(
            analysis["metrics"]["active_listening_count"],
            DEFAULT_PHASE_CONFIG["active_listening_cap"],
        )

    def test_e_in_vs_e_ex_detection(self):
        transcript = [
            {"speaker": "student", "text": "Goedemiddag, ik ben Lisa."},
            {"speaker": "student", "text": "We hebben een gesprek over uw klachten."},
            {"speaker": "patient", "text": "Ik slaap slecht en maak me zorgen."},
            {"speaker": "student", "text": "Wat bedoelt u precies met slecht slapen?"},  # E-in
            {"speaker": "patient", "text": "Ik word vaak wakker."},
            {"speaker": "student", "text": "Hoe voelt dat voor u als u wakker wordt?"},  # E-in follow-up same topic
            {"speaker": "patient", "text": "Het is vermoeiend."},
            {"speaker": "student", "text": "Naast uw slaap wil ik ook vragen naar uw voeding."},  # E-ex
            {"speaker": "patient", "text": "Ik eet weinig."},
            {"speaker": "student", "text": "Kunt u een voorbeeld geven van wat u gisteren at?"},  # E-in again
            {"speaker": "student", "text": "Dank u wel voor uw antwoorden."},
        ]
        analysis = analyze_conversation_phases(transcript, DEFAULT_PHASE_CONFIG)
        self.assertGreaterEqual(analysis["metrics"]["e_in_count"], 2)
        self.assertGreaterEqual(analysis["metrics"]["e_ex_count"], 1)
        self.assertEqual(analysis["phases"]["phase2"]["items"]["e_in_deepen"]["present"], True)
        self.assertEqual(analysis["phases"]["phase2"]["items"]["e_ex_shift_topic"]["present"], True)

    def test_gordon_detection_counts(self):
        transcript = [
            {"speaker": "student", "text": "Gebruikt u medicatie of heeft u allergie?"},
            {"speaker": "patient", "text": "Ik gebruik geen medicatie meer."},
            {"speaker": "student", "text": "Hoe is uw voeding en drinken de laatste tijd?"},
            {"speaker": "patient", "text": "Ik eet weinig en voel me vaak moe."},
            {"speaker": "student", "text": "Hoe slaapt u, komt u de nacht door?"},
            {"speaker": "patient", "text": "Slecht, ik word vaak wakker."},
            {"speaker": "student", "text": "Maakt u zich zorgen, ervaart u stress?"},
        ]
        gordon = detect_gordon_patterns(transcript, DEFAULT_PHASE_CONFIG)
        self.assertGreaterEqual(gordon["covered_count"], 3)
        self.assertEqual(gordon["total_patterns"], 11)
        self.assertGreater(gordon["coverage_percent"], 0)

    def test_gordon_word_boundary_no_false_positive(self):
        transcript = [
            {"speaker": "student", "text": "Ik weet het nog niet zo goed, maar verder niets over eten."},
        ]
        gordon = detect_gordon_patterns(transcript, DEFAULT_PHASE_CONFIG)
        self.assertEqual(gordon["covered_count"], 0)
        self.assertEqual(gordon["coverage_percent"], 0)

    def test_active_listening_excludes_questions(self):
        transcript = [
            {"speaker": "student", "text": "Hoe gaat het?"},
            {"speaker": "patient", "text": "Goed."},
            {"speaker": "student", "text": "ja"},
        ]
        analysis = analyze_conversation_phases(transcript, DEFAULT_PHASE_CONFIG)
        # Only the backchannel "ja" should count, not the question.
        self.assertEqual(analysis["metrics"]["active_listening_count"], 1)


if __name__ == "__main__":
    unittest.main()
