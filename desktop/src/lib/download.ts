// 统一的浏览器下载触发 —— 收敛各面板里重复的
//   Blob → createObjectURL → a.click → revokeObjectURL 样板。
//
// 旧实现同步 `a.click(); URL.revokeObjectURL(url);` 在 WebView2 / 部分浏览器
// 下可能在上传递交前就把 object URL 回收掉, 偶发下载失败。这里统一改为
// 延迟到微任务之后再 revoke, 一处修复全链路受益。

/** 传入一个已构造好的 Blob 与文件名, 触发浏览器下载. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // 交给微任务队列尾部再回收, 确保浏览器已开始读取 object URL.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** 便捷封装: 把 JSON 对象序列化后导出. */
export function downloadJson(data: unknown, filename: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: 'application/json',
  });
  downloadBlob(blob, filename);
}

/** 便捷封装: 导出纯文本/任意文本态内容 (markdown/csv/txt). */
export function downloadText(content: string, filename: string, mime = 'text/plain'): void {
  downloadBlob(new Blob([content], { type: mime }), filename);
}