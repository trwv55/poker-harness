# Автосохранение vision и подключение агента

Статус: **реализация и независимая приёмка завершены; разрешён локальный выпуск**, 2026-11-20.
Offline-проверки и два реальных agent follow-up на CoreWeave приняты; миграция0018,
backup и runtime SHA проверяются при выпуске по runbook. S3 adapter остаётся будущим срезом.
Предыдущий draft о временном контексте заменён целиком: владелец выбрал
автоматическое постоянное хранение и включение в историю/статистику.
Нет TTL руки, временного scope, Save UI или отдельного действия сохранения.

Основания: [CLAUDE](../../../CLAUDE.md), [архитектура](../../../.claude/ARCHITECTURE.md),
[позиционирование](../../../.claude/POSITIONING.md), [UX](../../../.claude/SESSIONS_UX.md),
[evals](../../../.claude/EVALS.md), [политика публикации](../../publishing-policy.md),
[vision-срез](2026-11-16-pokerdom-vision-reading-design.md), §§5–9,19–20.
Новая политика явно заменяет прежний default-no-history для успешных vision-рук;
остальные ограничения чтения/расчётов/приватности остаются. Изученный код —
ebf39fa61260f92ef97cebdc851cc1b97c061618.

## 1. Цель и текущая проблема

Сейчас Pokerdom Telegram worker читает/проверяет таблицу и печатает рассказ
с фишками и ББ, но job имеет session_id=NULL и не создаёт hands/analyses.
Follow-up агент не получает именно эту руку через sessions.last_analysis_id.
Отдельный профильный pokerdom_nickname уже хранится в БД, не равен gg_nickname.

Нужно автоматически записывать успешно прочитанную полную проверенную руку
в существующую историю, получать настоящий кодовый AnalysisResult и подключать
его к обсуждению обычной активной сессии. Пользователь не управляет сохранением.
Скрин остаётся источником-гипотезой; сохранение не превращает его в румовую HH.
Семь сервисов и границы чистых пакетов сохраняются. Нет нового vision table,
технической сессии, новых моделей, формул, румов или live-подсказок.
Подготовка к S3 через порт хранения обязательна уже в следующей реализации (§8).
Первый будущий срез предусматривает local adapter; сам S3-сервис/SDK пока не подключаются.

## 2. Поток и условия записи

```text
admission/ник → обычная активная сессия + job
 → прежние format/read/binding/replay/validate
 → сохранение source + raw/canonical/готовый enriched в hands
 → analyze_hand готового enriched существующим кодовым pipeline
 → настоящий AnalysisResult + decision_points в analyses
 → owner/current-session publication last_analysis_id
 → прежний friendly рассказ; follow-up и обычная история/статистика
```

Допуск в аналитическую историю: read_status=ok, complete supported action log,
replay_status=passed, однозначный герой, обязательные поля/контрольные банки
сверены, raw/canonical/enriched присутствуют. Неполнота, неоднозначный ник,
неподдерживаемый формат, провал legality/банка не создают verified Hand/Analysis.
Сырые ответы/ошибки остаются служебными job/trace/file artifacts, не руками истории.

EnrichedHand повторно не проигрывается ради analysis_id. Используется обычный
analyze_hand через существующий process-pool helper worker.pipeline
_analyze_hand_with_cache. Сохраняется настоящий AnalysisResult, даже если ядро
не смогло судить ни одной точки; пустая заглушка ради ID запрещена.
AnalysesRepo.save получает реальные report.decision_points для колонок контекста.
Не включать старый screenshot_analyze путь без адаптации его side effects.

Если анализ временно упал после валидного чтения: сохранить валидный hand как
незавершённый существующий checkpoint, показать рассказ и понятный статус
недоступности обсуждения; не публиковать analysis pointer и не выдавать заглушку
за анализ. Возобновить с готового enriched без повторного OCR/replay. Исторические
аналитические показатели по точкам требуют готового Analysis; частоты действий
допустимы только для валидного canonical. Ошибочный OCR не маскируется этим путём.

## 3. Существующая схема и минимальные дополнения

Подтверждённые поля:

- hands: session_id NOT NULL FK sessions, tournament_id nullable FK tournaments,
  provenance, image_hash, raw/canonical/enriched JSON, schema_version.
- RawHand.source_ref уже существует; RawHand.vision.image_hash тоже.
  HandsRepo.save_raw переносит vision.image_hash в hands.image_hash.
- analyses.hand_id/result и отдельные decision_points уже существуют.
- sessions.last_analysis_id — FK analyses ON DELETE SET NULL; active_spot_id
  отдельный указатель. История агента хранится в sessions.agent_history.

В raw.source_ref хранится переносимый logical backend/key ref, SHA — в
vision.image_hash/hands.image_hash. Абсолютный путь — только legacy input для
локального resolver, не новый долговременный ref. Сопутствующий versioned
reading/result manifest адресуется через тот же storage backend и
содержит наблюдения, checks, conditional outcome и версии схемы/промпта.
Не вводить storage_scope/expires_at, новый owner/vision table или обязательные
поля старых JSON. Для provenance/read metadata использовать существующие поля
и sidecar; если понадобится новый JSON-ключ, он только optional/backward compatible.

Неподтверждённый tournament FK — NULL, не новая фиктивная Tournament. Отсутствующий
hand_no остаётся отсутствующим, timestamp — прежний неизвестный sentinel,
не «сейчас» как якобы время игры. Путь файла/время загрузки — технические данные,
не номер/дата раздачи рума. Временные окна нынешних статистических репозиториев
относятся к sessions.started_at: это окно загрузки/разбора, не доказанная дата игры.

Новые DB-колонки для этой политики не обоснованы. Возможен индекс image_hash
для поиска дубликатов, только после проверки existing schema/query plan.
Уникальный owner+hash constraint нельзя обещать обычным индексом на Hand:
owner сейчас выводится через Session. Минимальная сериализация описана ниже.
Миграция нужна лишь для действительно выбранного индекса/изменения ограничения;
её необходимость и SQL должны быть показаны на ревью, а не предполагаться.

## 4. Идемпотентность и повтор того же изображения

Ключ минимального дедупа — **владелец + SHA256 точных исходных байтов** в
поддерживаемом screenshot-пути; поиск через Hand.image_hash JOIN Session.owner.
Разные владельцы не объединяются. Crop/сжатие/повторное фото дают другие байты:
распознавание «той же раздачи» по ним вне среза. Это не дедуп HH и screenshots.

Один общий repo save/resume entrypoint для vision берёт PostgreSQL transaction
advisory lock по owner+hash, затем ищет существующий hand, создаёт либо возвращает
его и атомарно записывает checkpoint job.hand_id/payload. Все vision writers,
включая legacy screenshot entrypoint, должны использовать эту границу либо
оставаться недостижимыми. Не держать транзакцию/лок во время LLM или анализа.
Concurrent duplicate calls после lock видят один committed Hand; исключение
не должно оставлять «созданный hand, потерянный payload» без lookup восстановления.
Это прикладной инвариант под общей границей, не универсальный unique constraint
для произвольного SQL в обход репозитория.

Аналогично под hand lock: проверить готовый Analysis, не повторять save и
insert decision_points на retry. Запись analysis+points и checkpoint связаны
транзакцией; публикация pointer и отправка восстанавливаются отдельно.
Повтор точно того же файла переиспользует IDs и сохранённый успешный результат,
не увеличивает counts и не делает новый OCR автоматически.

Техническое предложение для пограничного случая после /new: повтор тех же
байтов тем же владельцем переиспользует исходные Hand/Analysis IDs и исходный
Hand.session_id, не переносит руку в новый вечер. Новая Session.last_analysis_id
может указывать на этот ранее сохранённый анализ как на текущую обсуждаемую
руку. Cross-session доступ разрешён только после проверки same owner при
активации и каждом tool call; номер чужой руки доступа не даёт. Готовый успешный
source/analysis переиспользуется без нового LLM-чтения/его оплаты; обычные
проверки доступности source/checkpoints сохраняются, owner/данные не затираются.
Историческая принадлежность и session stats определяются оригинальным
Hand.session_id: discussion pointer не означает вторую сыгранную руку и не
увеличивает counts нового вечера. Summary явно различает исходную сессию руки
и текущую сессию обсуждения, не приписывает игру новой сессии.

Новая исправленная транскрипция/герой для тех же байтов не затирает старую тихо.
Сохранить исходный reading и версии исправлений в manifest; сверить owner/target,
отметить причину изменения. Не переиспользовать анализ с другой трактовкой героя.
Явная коррекция может инвалидировать downstream checkpoints через контролируемый
replace_raw и пересчёт изменённой руки; это не обычный retry и не новая копия Hand.
Не менять карту/сумму ради успешного replay. При конфликте исправлений остановить
перезапись и дать диагностику; полный UI редактора не входит в этот срез.

## 5. Обычная сессия и текущая рука

Новый vision intake использует SessionsRepo.active_or_create, как остальные
пользовательские входы. Создаётся вечер, не технический контейнер. Новый job
несёт session_id; поддержка старых session_id=NULL jobs остаётся: явное
однократное связывание при восстановлении без перепривязки уже начатой руки
в другой вечер. Нельзя автоматически присоединить поздний job к новой /new сессии.

| Событие | Семантика current context |
|---|---|
| Принят новый скрин после admission | Привязать к текущему вечеру; pending intake/job однозначен. Снять старый pointer, чтобы follow-up не говорил о прежней руке во время нового чтения. |
| Успех hand+analysis | Опубликовать analysis_id только для всё ещё актуального выбора в открытой сессии этого владельца; active_spot_id очистить. |
| Новый скрин провалился/отменён | Не активировать прежнюю руку молча. Ранее сохранённая рука остаётся в истории и доступна явным выбором. |
| HH scan принят | Снять текущий pointer; его позднее завершение не должно стереть более свежую выбранную руку. |
| Deep-dive выбран | Проверить owner/target; после готового анализа публиковать только если выбор ещё актуален. |
| Точные повторные байты после /new | Новая сессия обсуждает тот же owned analysis через свой pointer; Hand.session_id/IDs и статистическая принадлежность не меняются (§4). |
| Новый текстовый spot | Существующий active_spot_id/его version становится текущим, не смешивается с analysis. |
| /new | Новый вечер без текущей руки/истории диалога; старые сохранённые руки остаются в прежней сессии. Профильный ник сохраняется. |

Индекс uq_jobs_player_id_running сериализует running jobs одного игрока;
claim учитывает priority, не только время. active_or_create имеет advisory lock.
Эти факты не защищают сами по себе от /new и новых intake/callback между началом
и завершением старого job. Безусловный set_last_analysis сегодня такой защиты
не имеет. Нужен publish-if-current под согласованным owner/session lock и
worker fencing, проверяющий session closed/current и актуальный контекстный выбор.

Минимальное предложение без новой колонки: атомарно связывать admission/job с
сессией под owner lock; использовать сохранённый job ID и поиск последнего
принятого context-selecting job в этой сессии (vision/HH/deep-dive), плюс проверку
active_spot_id/version. Порядок выбора задаётся сериализованной DB-регистрацией,
не временем завершения или догадкой по глобальному ID. Вопрос агенту сам по себе
не выбирает новую руку. Все writers указателей, включая /new/spot, участвуют
в одном протоколе. Точный SQL/CAS доказать race tests до реализации публикации.
Если текущие поля этого не обеспечат, отдельный selection-token — **техническое
предложение для ревью с обоснованием**, не заранее требуемая новая DB-колонка.
Долгий расчёт не удерживает DB lock. locked_by fencing не заменяется current-check.

## 6. Агент, ownership и grounding

Готовый AnalysisResult подключается через существующий sessions.last_analysis_id.
Calcs._analysed_hand → AnalysesRepo.enriched_hand уже проверяет owner через Session.
Но worker._agent_context сейчас получает analysis/hand по ID без такой проверки:
перевести его на owner/session-aware resolver, не считать внутренний ID capability.
Для reused duplicate из §4 resolver различает owned origin session руки и
current discussion session: их IDs могут не совпадать. Требуется один и тот же
владелец обеих сессий и актуальный pointer/выбор в сессии обсуждения, а не
искусственное равенство Hand.session_id=current session_id. Прямой доступ по
analysis_id вне разрешённого выбора по-прежнему закрыт.
Проверка принадлежности нужна и при tool call, и при публикации ответа после /new
или смены выбранной руки. Удалённая/недоступная цель — понятный отказ, не fallback
на произвольную последнюю руку другого вечера.

opponent_range/equity_vs_range получают прежний analysis_id и готовый canonical.
Математика и прежние missing-chart/depth/multiway/KO/range ограничения не меняются.
AnalysisResult без судимых решений не означает «сыграно верно». Зона доверия
задаётся инструментом/допущением, не самим фактом сохранения screenshot.

Summary агента строится кодом из AnalysisResult/enriched и source/checks:
screenshot-гипотеза, неизвестные поля, conditional visible-card outcome.
Не передавать ложную независимую room payout сверку. В agent/history.py уже есть
HAND_KEY=analysis_id; при смене руки не переносить её narrowings, summary и
разрешённые числа в новую. Не удалять саму сохранённую руку при /new.
Проверки allowed numbers/assumptions/ответов сохраняются; LLM выбирает расчёты,
не выполняет арифметику и не объявляет успешный replay доказательством OCR.

## 7. История и статистика: инвентарь интеграции

Руки сохраняются автоматически и **входят** в обычную историю и статистику.
Этот список — coverage проверки подключения, не фильтр исключения context.
Никакого нового временного scope в запросах нет.

| Текущий entrypoint | Что проверить при интеграции |
|---|---|
| HandsRepo.save_raw/save_canonical/save_enriched/get/_get_row | Правильный owner через session, image_hash/source_ref, валидные checkpoint stages и один ID на повтор. |
| AnalysesRepo.save/_save_points/get_by_hand/enriched_hand | Настоящий result/точки, один набор на hand, owner resolution. Неполный analysis не публикуется. |
| memory.repos._points_of → LeaksRepo.overview/coverage/coverage_and_cost/by_type | Vision points включаются ровно один раз, с прежними judged/mismatch/priced/assumption правилами и правильными знаменателями. Не считать zero placeholder «строгой» оценкой. |
| SessionsRepo.summary/_session_loss_bb | Distinct hand count, coverage/leaks/loss включают новую сохранённую руку после соответствующего stage; no extra Tournament count. |
| HandsRepo.player_canonical → calcs._measure/_hero_frequency/_opponent_frequency/_threshold | Уже допускает скрины без турнира: action-derived hero/field frequencies и all-in opportunities получают их один раз. Owner/window/position фильтры прежние. |
| HandsRepo.player_hands_by_tournament, canonical_by_tournament/list_by_tournament/count_by_tournament | Screenshot с tournament_id=NULL не попадает в чужую группу. Не выдумывать tournament для межтурнирного PlayerStats. |
| worker._tournament_stats и analysis.player_stats | Частоты из подтверждённых actions; исключение разбираемой руки/порог20 сохраняются. Screenshot без FK не приписывается турнирному сопернику. |
| HandsRepo.find_by_hand_no/find_session_by_hand_no | Пустой hand_no нельзя использовать как уникальную цель. HH lookup/deep-dive/ranges/disagree должны выбирать screenshot по явному owned ID, а не совпавшему пустому номеру. |
| HandsRepo.last_canonical → bot.handle_alias_command | Новая последняя screenshot не должна вести к фиктивной GG alias-сшивке по отсутствующему room tournament. Отказ/подсказка, не fake IDs. |
| TournamentsRepo.player_scan_summaries/save_scan_summary | Скрин не переписывает HH scan_summary; session/global hand counts могут увеличиться, summary самого HH-файла остаётся о файле. |
| OpponentsRepo.get_or_create/link/list_for_player/links, NotesRepo | Ни P1, ни одинаковый nickname не склеивают личности/румы. Owner из Session остаётся изоляцией заметок и связей. |
| worker._agent_context, calcs._analysed_hand, routing tools_for/_perform | Current owned analysis для hand-tools; history tools остаются обычными статистическими запросами. Нет скрытого перехода к другому analysis. |
| CalcCacheRepo, agent history | Чистые eqc/eqr/equity_mc кэши по математическим входам переиспользуются. Исторические cached aggregates (если есть/будут) инвалидируются/ключуются изменением набора history; не вводить новый кэш заранее. Tool history не является проверкой доступа по ID. |

Observed действия, позиции, обязательные взносы и суммы дают action-derived
частоты. В screenshot raw.collected/summary не дописываются рассчитанные деньги,
showdowns не фабрикуются из видимых карт. Производные pot/net/end остаются
calculated/conditional/unknown. Save не делает их подтверждённой доходностью,
реальным payout или наблюдаемым showdown для outcome-зависимых stats.
Для каждого показателя использовать его прежнюю capability/coverage, исключая
неподтверждённое outcome именно из его знаменателя, не исключая всю руку из
частот действий. Если существующий показатель не умеет такую границу, не
публиковать его как достоверный для screenshot до явной адаптации/теста.
Новый winnings/ROI расчёт в этот срез не входит.

Нынешний opponent_links ключ — owner+room_tournament_id+participant_label,
не универсальная room-scoped nickname identity. Поддержанный формат не доказывает
бренд. Source nicknames сохраняются в vision metadata, но не создают автоматическую
GG-сшивку. Новую универсальную identity schema не строить ради этого среза;
действующие tools без надёжной room/participant связи отказывают в персональном
opponent-aggregate, а hero/field action counts продолжают работать.

## 8. Исходники: обязательный порт хранения, local сейчас / S3 позже

**Принято владельцем:** разрабатывать сразу с возможностью быстро подключить S3.
Это обязательная архитектурная закладка следующей реализации, не включённый S3.
Узкий SourceStorage порт (рабочие имена):

- put(owner_id, bytes, MIME, operation_key) → SourceRef: идемпотентная запись
  исходных байтов; повтор операции не создаёт другой source.
- read(owner_id, SourceRef, max_bytes) → bytes либо managed local temporary file.
- delete(owner_id, SourceRef, expected_hash) → идемпотентный результат удаления;
  координатор до вызова проверяет DB refs/активные jobs и право на cleanup.

SourceRef — backend + opaque object key + SHA256/size/MIME metadata, не публичный
URL и не host absolute path. Owner хранится/проверяется в БД через принадлежащую
ему сессию/руку; знание ref/key не даёт доступ. Keys — random UUID (допустим
opaque owner-prefix), без реальных ников/Telegram IDs. Owner+image-hash дедуп
из §4 остаётся логическим: cross-owner shared-file дедуп сейчас не нужен.

LocalSourceStorage реализуется первым на существующем DATA_DIR shared volume
bot/worker; физические пути и проверка symlink/path escape скрыты внутри adapter.
Текущие UUID intake files можно зарегистрировать без лишнего копирования/rename.
Новые jobs и hands используют logical ref; worker получает bytes/tempfile через
внедрённый порт, проверяет размер/SHA/MIME до модели. Poker parser получает те же
bytes, engine — те же контракты; storage/SDK/platform imports в pure packages
запрещены. Проводка bot admission/worker/file lifecycle — разрешённая инфраструктура.

Existing source_ref/image_hash переиспользуются: например, source_ref кодирует
backend/key строкой, остальная metadata — existing manifest/optional compatible
JSON. Малое optional backend/key metadata допустимо, если фактическая сериализация
требует; новая таблица/DB-колонка без обоснования не нужна. Legacy held jobs с
source_file и сохранённые raw.source_ref absolute paths читает локальный resolver:
валидирует managed root/owner/hash, возвращает logical ref. Не переписывать все
старые данные/схемы; обновить checkpoint при обычном успешном восстановлении.
Нельзя принять произвольный старый путь извне как разрешение читать файл хоста.

DI SourceStorage задаётся bot/worker deps; config selector local|s3, default local.
S3SourceStorage добавляется позже: SDK, endpoint/bucket/server-env credentials
только в adapter/config. Выбор s3 без доступного adapter/config явно ошибочен,
не скрытый fallback на диск. Подключение не меняет poker pipeline/LLM контрактов,
hand tools и UI; read может материализовать temporary file внутри adapter.
Сейчас эта спецификация не устанавливает SDK/credentials и не создаёт bucket.

Source сохраняет оригинальные байты, не BLOB в PostgreSQL и не пережатую копию.
Успешный source хранится с постоянной рукой без автоматического TTL. DB хранит
logical ref/hash/metadata; подписанная ссылка или временный local file не становятся
новым постоянным source_ref/job ref.

DB и file/object store не атомарны. Порядок: complete durable put + verified
metadata/manifest ready → owned Hand/checkpoint transaction → analysis/pointer.
operation_key и owner+hash lookup позволяют retry после crash до/после DB commit.
Для local put — temporary write + atomic rename внутри volume; для будущего S3 —
повторяемый put/проверка object metadata. Не удалять прежний source до подтверждения
ссылок/успеха. Missing source — диагностика восстановления, не вторая Hand.

Cancel/failed/orphan cleanup разрешён лишь без ссылок из Hand/history, manifest/
job и активного intake/worker. Проверку refs и cleanup claim сериализовать с
созданием новой ссылки; только затем port.delete. Retry delete отсутствующего
объекта успешен. При crash intent/operation checkpoint позволяет продолжить;
при сомнении файл/объект оставить и диагностировать. Cancel повторного снимка
не удаляет source уже сохранённой руки. Не трогать HH/range/private eval файлы.
Успешным рукам TTL не назначается; отдельная политика удаления постоянных рук
не проектируется этим срезом.

### 8.1. Production — предложение развёртывания, не выбранный сервис

Сейчас local volume poker-harness_hh_files → /data; один VPS также может
использовать диск с отдельным off-host backup. Для внешних пользователей и
нескольких worker-хостов предложен external private S3-compatible backend,
не MinIO на том же VPS как якобы off-host защита. Вендор/цена/договор не выбраны.
Bucket private, выдача только после owner-check через bot delivery либо короткую
подписанную ссылку. Credentials server-env, не prompt/logs. До внешней нагрузки
нужны file/LLM/storage limits, не новая тарифная система.

Backup обязателен для PG **и** source backend: pg_dump не содержит байты volume/
objects. Нужны совместимый DB snapshot+manifest и отдельный file/object backup
или versioning с проверенным restore/hash/permissions; versioning само по себе
не заменяет проверку восстановления. См. [runbook§11](../../deploy-runbook.md#11-бэкап-и-восстановление).
Время включения S3 (сразу на VPS или перед внешними пользователями) ещё обсуждается;
готовность порта/local adapter уже принята и от этого решения не зависит.
Никаких новых пользовательских вопросов/Save UI для доступа к S3 не требуется.

## 9. Приёмка реализации

Актуальные результаты проверок — в приватном implementation handoff, не в старых логах. Public tests — независимая синтетика;
три owner-confirmed приватных кейса/images/stories — offline integration fixtures,
не source для публичных тестов и не общий vision-бенчмарк.

| ID | Критерий и доказательство |
|---|---|
| C1 | Screenshot→валидный Hand+real Analysis/points→current pointer→follow-up подходящему tool. Используется тот же enriched, второй replay отсутствует; missing chart/multiway/KO дают прежнюю честную capability. |
| C2 | Обычная history/session/stats видит одну руку и один набор points; все entrypoints §7 покрыты, включая _points_of/player_canonical/denominators/loss. Никакой Save UI/команды согласия не требуется. |
| C3 | Retry/crash/concurrent duplicate exact owner+image bytes дают те же IDs, counts1 и существующий source с правильным SHA. Отдельно upload→/new→те же байты: Hand/Analysis IDs и origin Hand.session_id прежние, новый current pointer установлен, общая history count1, session membership/counts исходные, нового LLM-вызова нет. Проверить запрет другому владельцу на активацию и каждый tool call; другой crop не обещается дедуплицированным. |
| C4 | Ambiguous hero/incomplete/invalid/replay/bank failure не создают verified history/placeholder Analysis. Служебный artifact доступен для диагностики, не включён в статпоказатели. Analysis failure resume без OCR/replay; одна запись points. |
| C5 | /new/new image/HH scan/deep-dive/text spot/старый finish не перетирают новый current. Owner/session resolver и worker fencing проверены отдельно; профильный ник переживает /new. |
| C6 | tournament FK NULL, hand_no/time неизвестны без source. Ни fake Tournament, ни timestamp загрузки как время игры. Старые JSON/HH и nullable legacy jobs совместимы. |
| C7 | Screenshot provenance/conditional cards/outcome сохраняются; calculated payout не попадает в source collected/showdown/winnings. Action stats включаются, неподтверждённый outcome не загрязняет свои числитель/знаменатель. |
| C8 | Room/nickname/owner boundaries: нет межрумной/межвладельческой сшивки, fake alias IDs или утечки заметок; честный отказ tools без identity/capability, grounding/allowed numbers/assumptions сохранены. |
| C9 | SourceStorage fake и LocalSourceStorage проходят один контракт без AWS/network: logical ref, put/read/delete, bytes/SHA/size/MIME, owner/path guards, idempotent retry и legacy paths/jobs. Worker не предполагает local path; замена backend требует только adapter/config. Crash до/после storage/DB recoverable; cancel/cleanup не удаляет referenced history/active-job source. Backup PG+source restore проверен; одного pg_dump недостаточно. |
| C10 | Existing HH scan/JSON/lookup/ranges/history и chips+BB friendly view не регрессируют. Если обоснована migration/index — real PostgreSQL up/down/defaults/index/FK checks, без нового scope/TTL. |
| C11 | Privacy/package boundaries: новые DB координации только bot/worker/memory/agent/calcs; no infrastructure imports в pure packages, no private hands/prompts/images/reports в Git. |

C1/C4/C7 развивают прежние D1–D9/T1/T6; C2/C3/C5/C8 задают новое поведение D11
и T2/T4/T5; C11 — D12. Прежний D10 accuracy не переносится на follow-up модель.
Mock tools доказывают проводку, но не верность реальной agent routing/grounding:
её отдельная приватная model-eval приёмка требует отдельного запуска/бюджета.
Не повторять paid vision-группу ради DB проверки.

Будущие проверки: targeted real-PG integration/concurrency/FS recovery,
полный pytest -q -ra с явными skips/deselected и HH grid, Ruff/Pyright по правилам
проекта для pipeline/DB changes. Старые зелёные логи не подтверждают новый код.
В реализации выполняются offline тесты на disposable PostgreSQL; live DB/API не используются.

## 10. Исполнение и передача Luna

Блокирующего вопроса владельцу о сохранении/TTL/save UX больше нет: последнее
решение однозначно — автоматическая постоянная история. Остаточные технические
выборы (нужен ли индекс, SQL publish-if-current, детали регистрации legacy UUID source)
исполнитель обосновывает по коду и tests; они не требуют придуманного нового
продуктового permission-flow. Production storage timing остаётся предложением
§8.1, не блокирует однозначность принятой политики auto-history.

Срезы: (1) SourceStorage/local adapter + DI/legacy resolver, общий owned
image-hash save/resume и source safety/session binding;
(2) reuse enriched→real analysis/points, current pointer/agent integration и
coverage обычной истории; (3) race/recovery/identity/outcome/backup regression.
Не выпускать автозапись раньше корректного идемпотентного stats/identity поведения.

Sol — реализация и собственные проверки; Luna — независимый read-only SDD review;
root — координация. Astra не нужна для этой постановки. Будущее ownership:
memory models/repos и только обоснованная migration/index; platform SourceStorage
port/LocalSourceStorage и DI/config, future S3 adapter отдельно; bot intake, worker
pokerdom/pipeline/current publication; agent history/explanation summary/calcs
resolver; source file policy/config/docs/synthetic integration tests.
Чистые парсеры/normalizer/engine/стратегическая математика не меняются без доказанной
необходимости. Дополнительной vision таблицы нет; storage port/local adapter обязательны,
внешний S3-сервис пока не включается. Тесты fake/local — без AWS/network.

Пакет Luna: полный этот draft + последний brief владельца, предыдущая vision
спека/CLAUDE/архитектура/позиционирование/EVALS/UX/privacy и actual code inventory
§7. Ответ — findings с конкретными сценариями, затем C1–C11 «подтверждён/нарушен/
не проверен». Implementation handoff содержит точные public/new/private files,
base/source SHA, diff, команды/exit/skips, real-PG race/checkpoint/backup evidence
и приватные артефакты. Живой runtime пока прежний, без auto-history; локальная
реализация ждёт независимого ревью и отдельного release-поручения.


### Реализационное решение 2026-11-19

Выбран `sessions.context_revision` (0018, default0) вместо вывода current выбора
из job ID: pending nickname ещё не имеет job, text spot и /new меняют выбор
отдельно от очереди, priority не равен порядку выбора. Admission увеличивает
revision и снимает pointer; job хранит snapshot, publish делает owner/open/revision
CAS. Text spot и закрытие вечера также меняют revision. Historical HH deep-dive
закрытого вечера сохраняет прежнюю возможность прочитать разбор, но не активирует
его в закрытой/чужой текущей сессии. Legacy job без revision не публикует current
молча. Индекс hands.image_hash ускоряет owner-joined lookup; уникальность обеспечена
общим owner+hash advisory lock, не обещана индексом на отсутствующий owner столбец.
`VisionMeta.reading_ref` optional: portable source manifest с исходным result и
версиями request; старые JSON читаются без него. SourceStorage local по умолчанию,
SOURCE_STORAGE_BACKEND=s3 явно отказывает до появления адаптера. Port и fake/local
контракты реализуются без S3 SDK. Полный evidence —
`reports/pokerdom-implementation/vision-autosave-implementation-handoff.md` (private).
