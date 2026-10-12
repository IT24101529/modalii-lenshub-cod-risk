# Data provenance

The group reports that MODALII LENSHUB provided 4,316 order records. The dataset-preparing member explains that delivery and return reports were matched on Waybill Number: returned matches are labelled 1 and successful/non-returned deliveries are labelled 0. This replaces the initial proposed collection-based target.

Original exports and merge code were not supplied for independent verification. Return-file completeness and successful versus pending/unmatched outcomes need source-level confirmation.

The modelling CSV removes waybill identifiers, phone-derived prefixes and unvalidated address-derived city text. It retains all 206 zero-COD orders, whose business meaning remains unconfirmed. A sequential record ID is used for audit only.

Dataset SHA-256: `78283eea4de4e4c4732de84a0cbd4c0d2da3c7f903687796abe18a6f1b96a96b`.

Lecturer approval was confirmed by the group. Retain client permissions and approval evidence privately. The repository intentionally excludes the client dataset and row-level predictions.
