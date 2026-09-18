import { copyFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const target = resolve('dist/research');
mkdirSync(target, { recursive: true });
for (const name of ['schema.sql', 'lifecycle-schema.sql', 'dashboard.html']) {
  copyFileSync(resolve('src/research', name), resolve(target, name));
}
