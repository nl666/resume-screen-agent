from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from resume_screen_agent.redact import redact_basic_personal_info, redact_sensitive_info


class RedactTests(unittest.TestCase):
    def test_redacts_common_personal_identifiers(self) -> None:
        text = """南亮
姓名：南亮
手机号：+86 138-1234-5678
邮箱：nanliang@example.com
微信：nl_agent_2026
身份证：11010119900307123X
地址：北京市海淀区中关村大街 1 号
毕业院校：北京大学
证书编号：CERT-2024-ABC123
"""

        redacted = redact_sensitive_info(text)

        self.assertNotIn("南亮", redacted)
        self.assertNotIn("138-1234-5678", redacted)
        self.assertNotIn("nanliang@example.com", redacted)
        self.assertNotIn("nl_agent_2026", redacted)
        self.assertNotIn("11010119900307123X", redacted)
        self.assertNotIn("北京市海淀区", redacted)
        self.assertNotIn("北京大学", redacted)
        self.assertNotIn("CERT-2024-ABC123", redacted)
        self.assertIn("[NAME_REDACTED]", redacted)
        self.assertIn("[PHONE_REDACTED]", redacted)
        self.assertIn("[EMAIL_REDACTED]", redacted)
        self.assertIn("[WECHAT_REDACTED]", redacted)
        self.assertIn("[ID_CARD_REDACTED]", redacted)
        self.assertIn("[ADDRESS_REDACTED]", redacted)
        self.assertIn("[SCHOOL_REDACTED]", redacted)
        self.assertIn("[CERTIFICATE_ID_REDACTED]", redacted)

    def test_redacts_internal_project_names_but_keeps_skill_evidence(self) -> None:
        text = """项目名称：Resume Screen Agent
内部项目：Apollo 智能风控平台
项目名称：智能客服中台（公司内部）
参与公司内部项目Orion，负责 RAG、Tool Calling、FastAPI、Redis。
"""

        redacted = redact_sensitive_info(text)

        self.assertIn("项目名称：Resume Screen Agent", redacted)
        self.assertNotIn("Apollo", redacted)
        self.assertNotIn("智能客服中台", redacted)
        self.assertNotIn("Orion", redacted)
        self.assertIn("[INTERNAL_PROJECT_REDACTED]", redacted)
        self.assertIn("RAG", redacted)
        self.assertIn("Tool Calling", redacted)
        self.assertIn("FastAPI", redacted)
        self.assertIn("Redis", redacted)

    def test_backward_compatible_function_uses_stronger_redaction(self) -> None:
        redacted = redact_basic_personal_info("姓名：张三\n微信：zhangsan_ai\n")

        self.assertIn("[NAME_REDACTED]", redacted)
        self.assertIn("[WECHAT_REDACTED]", redacted)


if __name__ == "__main__":
    unittest.main()
