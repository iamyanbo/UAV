/** A proposal/prior-art gate must not inherit the implementation validation gate. */
export function isProposalOnlyTask(task: { task_kind?: string; brief_md?: string } | string): boolean {
  const body = typeof task === "string" ? task : task.brief_md ?? "";
  if (/^Research stage:\s*(implementation|representative-evaluation)\s*$/im.test(body)) return false;
  return /proposal and prior-art gate, not an implementation or experiment task/i.test(body)
    || /proposal\/prior-art only[\s\S]{0,120}no implementation/i.test(body);
}

export function isMethodDevelopmentTask(task: { task_kind?: string; brief_md?: string }): boolean {
  return !isProposalOnlyTask(task)
    && (task.task_kind === "method-development"
      || /METHOD_DEVELOPMENT_REQUIRED|DESIGN AND IMPLEMENT METHOD/i.test(task.brief_md ?? "")
      || /^Research stage:\s*(implementation|representative-evaluation)\s*$/im.test(task.brief_md ?? ""));
}
