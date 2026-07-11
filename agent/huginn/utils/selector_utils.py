"""选择器自动生成 — 受 Scrapling SelectorsGeneration mixin 启发.

为 DOM 元素生成 CSS/XPath 选择器, 用于:
  - browser_tool 提取元素后自动生成可复用选择器
  - 知识库自动入库时记录元素定位方式
  - 网站改版后用选择器重新定位

Playwright/Selenium 版本: 接收 element handle 或 dict, 不依赖 lxml.
"""

from __future__ import annotations

from typing import Any


def generate_css_selector(element_info: dict[str, Any]) -> str:
    """从元素信息生成 CSS 选择器.

    element_info 应包含: tag, id, class, parent (递归结构)
    或者传入 playwright element handle 的 evaluate 结果.

    返回形如 "#content > div.list > article:nth-of-type(2)" 的选择器.
    """
    parts: list[str] = []
    current = element_info

    while current and isinstance(current, dict):
        tag = current.get("tag", "").lower()
        if not tag or tag in ("#document", "html", "#text"):
            break

        elem_id = current.get("id", "")
        if elem_id:
            parts.append(f"#{elem_id}")
            break  # id 唯一, 到此为止

        # 不用 class 定位 (多网站共享 class), 用 tag + nth-of-type
        parent = current.get("parent")
        if parent and isinstance(parent, dict):
            siblings = parent.get("children", [])
            same_tag_count = 0
            target_index = 0
            for sib in siblings:
                if isinstance(sib, dict) and sib.get("tag", "").lower() == tag:
                    same_tag_count += 1
                    if sib is current:
                        target_index = same_tag_count

            if same_tag_count > 1:
                parts.append(f"{tag}:nth-of-type({target_index})")
            else:
                parts.append(tag)
        else:
            parts.append(tag)

        current = parent

    return " > ".join(reversed(parts)) if parts else ""


def generate_xpath_selector(element_info: dict[str, Any]) -> str:
    """从元素信息生成 XPath 选择器."""
    parts: list[str] = []
    current = element_info

    while current and isinstance(current, dict):
        tag = current.get("tag", "").lower()
        if not tag or tag in ("#document", "html", "#text"):
            break

        elem_id = current.get("id", "")
        if elem_id:
            parts.append(f"//*[@id='{elem_id}']")
            break

        parent = current.get("parent")
        if parent and isinstance(parent, dict):
            siblings = parent.get("children", [])
            same_tag_count = 0
            target_index = 0
            for sib in siblings:
                if isinstance(sib, dict) and sib.get("tag", "").lower() == tag:
                    same_tag_count += 1
                    if sib is current:
                        target_index = same_tag_count

            if same_tag_count > 1:
                parts.append(f"{tag}[{target_index}]")
            else:
                parts.append(tag)
        else:
            parts.append(tag)

        current = parent

    return "//" + "/".join(reversed(parts)) if parts else ""


async def get_element_info(playwright_element: Any) -> dict[str, Any]:
    """从 Playwright ElementHandle 提取元素结构信息.

    返回递归结构: {tag, id, class, text, parent: {...}, children: [...]}
    用于 generate_css_selector / generate_xpath_selector.
    """
    # ponytail: 只取两层深度, 太深了选择器也没法用
    js = """(el) => {
        function info(e, depth) {
            if (!e || depth > 3) return null;
            const parent = e.parentElement;
            const children = parent ? Array.from(parent.children) : [];
            return {
                tag: e.tagName ? e.tagName.toLowerCase() : '',
                id: e.id || '',
                class: e.className || '',
                text: (e.textContent || '').substring(0, 100),
                parent: parent ? {
                    tag: parent.tagName ? parent.tagName.toLowerCase() : '',
                    id: parent.id || '',
                    children: children.map(c => ({
                        tag: c.tagName ? c.tagName.toLowerCase() : '',
                        id: c.id || '',
                    })),
                } : null,
            };
        }
        return info(el, 0);
    }"""
    try:
        return await playwright_element.evaluate(js)
    except Exception:
        return {}


def extract_indexed_state(
    elements: list[dict[str, Any]],
    max_items: int = 50,
) -> list[dict[str, Any]]:
    """把元素列表转成索引化的 state 列表 (BrowserAct 风格).

    每个元素: {index, tag, text, selector, attributes}
    agent 可以用 click(index) / input(index, text) 直接操作.
    """
    indexed: list[dict[str, Any]] = []
    for i, el in enumerate(elements[:max_items]):
        tag = el.get("tag", "")
        text = (el.get("text") or "").strip()
        attrs = el.get("attributes") or {}

        # 生成选择器
        el_info = {
            "tag": tag,
            "id": attrs.get("id", ""),
            "parent": None,  # state 模式不带 parent, 选择器会短一些
        }
        selector = generate_css_selector(el_info) or attrs.get("id", f"{tag}:nth-of-type({i+1})")

        indexed.append({
            "index": i,
            "tag": tag,
            "text": text[:80],
            "selector": selector,
            "is_link": tag == "a",
            "is_input": tag in ("input", "textarea", "select"),
            "is_button": tag == "button" or attrs.get("type") == "submit",
        })
    return indexed
