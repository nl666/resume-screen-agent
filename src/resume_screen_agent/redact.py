from __future__ import annotations

import re


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d[- ]?\d{4}[- ]?\d{4}(?!\d)")
ID_CARD_RE = re.compile(r"(?<![\dA-Za-z])(?:\d{15}|\d{17}[\dXx])(?![\dA-Za-z])")
WECHAT_LABEL_RE = re.compile(
    r"(?i)(微信(?:号)?|wechat|weixin|wx)(\s*[:：]?\s*)([A-Za-z][A-Za-z0-9_-]{4,24})"
)
NAME_LABEL_RE = re.compile(
    r"(?im)^(\s*(?:姓名|名字|真实姓名|name|candidate)\s*[:：]\s*)([A-Za-z][A-Za-z .'-]{1,40}|[\u4e00-\u9fff]{2,4})(\s*)$"
)
ADDRESS_LABEL_RE = re.compile(
    r"(?im)^(\s*(?:地址|现居地|居住地|通讯地址|联系地址|家庭住址|户籍地址|籍贯|所在地|所在城市)\s*[:：]\s*)(.+?)\s*$"
)
SCHOOL_LABEL_RE = re.compile(
    r"(?im)^(\s*(?:学校|毕业院校|院校|大学|高校|教育经历|最高学历院校)\s*[:：]\s*)(.+?)\s*$"
)
CERTIFICATE_NO_LABEL_RE = re.compile(
    r"(?im)^(\s*(?:证书编号|证书号|资格证编号|认证编号|certificate\s*(?:no\.?|number|id)?)\s*[:：]\s*)([A-Za-z0-9][A-Za-z0-9\-_/]{5,40})\s*$"
)
INTERNAL_PROJECT_LABEL_RE = re.compile(
    r"(?im)^(\s*(?:内部项目|保密项目|公司内部项目|项目代号|项目编号|内部系统名称|内部平台名称)\s*[:：]\s*)(.+?)\s*$"
)
INTERNAL_PROJECT_NAME_RE = re.compile(
    r"(?im)^(\s*项目名称\s*[:：]\s*)(?=.*(?:内部|保密|未公开|公司内部|私有))(.+?)\s*$"
)
INLINE_INTERNAL_PROJECT_RE = re.compile(
    r"(?P<prefix>(?:内部|保密|公司内部|未公开|私有)(?:项目|系统|平台|产品|工程|代号)\s*(?:[:：为是\-]\s*)?)(?P<value>[A-Za-z0-9_\-\u4e00-\u9fff]{2,40})"
)
PROJECT_CODE_RE = re.compile(
    r"(?i)(项目(?:代号|编号)|project\s*(?:code|id))(\s*[:：]?\s*)([A-Za-z0-9][A-Za-z0-9\-_/]{2,40})"
)

_SHORT_CHINESE_NAME_RE = re.compile(r"^[\u4e00-\u9fff]{2,4}$")
_COMMON_NOT_NAMES = {
    "简历",
    "个人简历",
    "求职简历",
    "教育经历",
    "工作经历",
    "项目经历",
    "技能清单",
    "个人信息",
    "联系方式",
    "自我评价",
}


def redact_basic_personal_info(text: str) -> str:
    """Backward-compatible wrapper for the stronger sensitive-info redactor."""
    return redact_sensitive_info(text)


def redact_sensitive_info(text: str) -> str:
    """Redact sensitive resume fields while keeping skill evidence intact."""
    text = _redact_labeled_and_topline_names(text)
    text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = PHONE_RE.sub("[PHONE_REDACTED]", text)
    text = ID_CARD_RE.sub("[ID_CARD_REDACTED]", text)
    text = WECHAT_LABEL_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}[WECHAT_REDACTED]", text)
    text = ADDRESS_LABEL_RE.sub(lambda match: f"{match.group(1)}[ADDRESS_REDACTED]", text)
    text = SCHOOL_LABEL_RE.sub(lambda match: f"{match.group(1)}[SCHOOL_REDACTED]", text)
    text = CERTIFICATE_NO_LABEL_RE.sub(lambda match: f"{match.group(1)}[CERTIFICATE_ID_REDACTED]", text)
    text = INTERNAL_PROJECT_LABEL_RE.sub(lambda match: f"{match.group(1)}[INTERNAL_PROJECT_REDACTED]", text)
    text = INTERNAL_PROJECT_NAME_RE.sub(lambda match: f"{match.group(1)}[INTERNAL_PROJECT_REDACTED]", text)
    text = PROJECT_CODE_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}[INTERNAL_PROJECT_REDACTED]", text)
    text = INLINE_INTERNAL_PROJECT_RE.sub(
        lambda match: f"{match.group('prefix')}[INTERNAL_PROJECT_REDACTED]",
        text,
    )
    return text


def _redact_labeled_and_topline_names(text: str) -> str:
    text = NAME_LABEL_RE.sub(lambda match: f"{match.group(1)}[NAME_REDACTED]{match.group(3)}", text)
    lines = text.splitlines(keepends=True)
    inspected = 0
    for index, line in enumerate(lines):
        if inspected >= 4:
            break
        stripped = line.strip()
        if not stripped:
            continue
        inspected += 1
        if _looks_like_topline_name(stripped):
            newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            lines[index] = f"[NAME_REDACTED]{newline}"
            break
    return "".join(lines)


def _looks_like_topline_name(value: str) -> bool:
    if value in _COMMON_NOT_NAMES:
        return False
    if _SHORT_CHINESE_NAME_RE.fullmatch(value):
        return True
    if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}", value):
        return True
    return False
