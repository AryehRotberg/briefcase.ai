from dataclasses import dataclass


@dataclass(frozen=True)
class Categories:
    # High criticality – direct, pervasive or sensitive personal‑data collection, tracking, sharing or sale
    high_criticality = [
        'Your personal data may be sold or otherwise transferred as part of a bankruptcy proceeding or other type of financial transaction',
        'Your personal data may be disclosed to comply with government requests without notice to you',
        'Your data is processed and stored in a country that is less friendly to user privacy protection',
        "This service does not honor 'Do Not Track' (DNT) headers",
        'Private messages can be read',
        'Your personal information is used for many different purposes',
        'This service gives your personal data to third parties involved in its operation',
        'This service shares your personal data with third parties that are not essential to its operation',
        'Your Personal data may be sold unless you opt out',
        'This service still tracks you even if you opted out from tracking',
        'Your personal or sensitive data is used for targeted advertising, including third-party and identity-based ads',
        'Your private content may be accessed by people working for the service',
        'This service tracks you on other websites',
        'Your browsing history can be viewed by the service',
        'Logs are kept for an undefined period of time',
        'Your location may be collected, used or shared, whether precisely or approximately, using IP address, GPS, or other device signals.',
        'Your biometric data is collected',
        'App required for this service requires broad device permissions',
        'You are tracked via web beacons, tracking pixels, browser fingerprinting, and/or device fingerprinting',
        'Information is gathered about you through third parties',
        'Many different types of personal data are collected',
        'Third-party cookies are used for advertising',
        'Your profile is combined across various products',
        'You are being tracked via social media cookies/pixels',
        'They store data on you even if you did not interact with the service',
        'Voice data is collected and shared with third-parties',
    ]

    medium_criticality = [
        'The copyright license maintained by the service over your data and/or content is broader than necessary.',
        'Tracking pixels are used in service-to-user communication',
        'Third-party cookies are used for statistics',
        'Content you post may be edited by the service for any reason',
        'Terms may be changed at any time',
        'A license is kept on user-generated content even after you close your account',
        'Your data may be processed and stored anywhere in the world',
        'Your content can be distributed through any media known now or in the future',
        'The court of law governing the terms is in a jurisdiction that is less friendly to user privacy protection.',
        'Critical changes to the terms are made without user involvement',
        'You must provide your identifiable information',
        'This service tracks which web page referred you to it',
        'Third party cookies are employed, but with opt out instructions',
        'This service assumes no liability for unauthorized access to your personal information',
        'Your personal data may be used for marketing purposes',
        'Extra data may be collected about you through promotions',
        'You must provide your legal name, pseudonyms are not allowed',
        'You are not allowed to use pseudonyms, as trust and transparency between users regarding their identities is relevant to the service.',
        'This service retains rights to your content even after you stop using your account',
        'Many third parties are involved in operating the service',
        'You can limit how your information is used by third-parties and the service',
        'The service may keep a secure, anonymized record of your data for analytical purposes even after the data retention period',
        'This service can use your content for all their existing and future services',
        'Prices and fees may be changed at any time, without notice to you',
        'Service fines users for Terms of Service violations',
        'The policy refers to documents that are missing or unfindable',
        'Your provided identifiable information is actively checked by the service',
        'You cannot opt out of promotional communications',
        'Some personal data may be kept for business interests or legal obligations',
        "First-party cookies used for site functionality or basic tracking",
        'Your information is only shared with third parties when given specific consent',
        'Your personal data is used for limited purposes',
        'Your personal data is aggregated into statistics',
        'No guarantee is given regarding quality',
        'Copyright license limited for the purposes of that same service but transferable and sublicenseable',
        'You waive your moral rights',
        "This service holds onto content that you've deleted",
        'Your content can be licensed to third parties',
        'If you offer suggestions to the service, they become the owner of the ideas that you give them',
        'If you offer suggestions to the service, they may use that without your approval or compensation, but they do not become the owner',
        'User-generated content can be blocked or censored for any reason',
        'The cookies used by this service do not contain information that would personally identify you',
        'This service takes credit for your content'
    ]

    low_criticality = [
        'Your personal data is not sold',
        'Your personal data is not shared with third parties',
        'IP addresses of website visitors are not tracked',
        'You are not being tracked',

        'There is a date of the last update of the agreements',
        'Alternative accounts are not allowed',
        'The service is transparent regarding government requests or inquiries that may involve your data.',
        'Information is provided about security practices',
        'You can request access, correction and/or deletion of your data',
        'The service deletes tracking data after a period of time and allows you to opt out',
        'User-generated content is encrypted, and this service cannot decrypt it',
        'The service claims to be GDPR compliant for European users',
        'The service is open-source',
        'You authorise the service to charge a credit card supplied on re-occurring basis',
        'The data retention period is kept to the minimum necessary for fulfilling its purposes',
        'Features of the website are made available under a free software license',
        'You can retrieve an archive of your data',
        'The court of law governing the terms is in a jurisdiction that is friendlier to user privacy protection.',
        'They may stop providing the service at any time',
        '30 days of notice are given before closing your account',
        'You have the right to leave this service at any time',
        "The service is provided 'as is' and to be used at your sole risk",
        'Any liability on behalf of the service is only limited to the fees you paid as a user',
        'You maintain ownership of your content',
        'The service has non-exclusive use of your content',
        'The service is only available in some countries approved by its government',
        'This service assumes no responsibility and liability for the contents of links to other websites',
        'Spidering, crawling, or accessing the site through any automated means is not allowed',
        'Your account can be suspended for several reasons',
        'When the service wants to make a material change to its terms, you are notified at least 30 days in advance',
        'Specific content can be deleted without reason and may be removed without prior notice',
        'The service has a no refund policy',
        'The service does not guarantee accuracy or reliability of the information provided',
        'This service is only available to users over a certain age',
        'User suspension from the service will be fair and proportionate.',
        'You can opt out of promotional communications',
        'You should revisit the terms periodically, although in case of material changes, the service will notify',
        'Your account can be deleted or permanently suspended without prior notice and without a reason',
        'You waive your right to a class action.',
        'You are responsible for maintaining the security of your account and for the activities on your account',
        'You are forced into binding arbitration in case of disputes',
        'Instructions are provided on how to submit a copyright claim',
        'User accounts can be terminated after having been in breach of the terms of service repeatedly',
        'You are entitled to a refund if certain thresholds or standards are not met by the service',
        'The service assumes no liability for any damages the user incurs',
        'You agree to defend, indemnify, and hold the service harmless in case of a claim related to your use of the service',
        'The service can intervene in user disputes',
        'The service informs you that its privacy policy does not apply to third party websites',
        'This service cannot be held responsible for disputes that you may have with other users',
        'No need to register',
        'This service is only available for use individually and non-commercially.',
        'You must report to the service any unauthorized use of your account or any breach of security',
        'The service claims to be CCPA compliant for California users',
        'You have a reduced time period to take legal action against the service',
        'This service will continue using anonymized user-generated content after erasure of personal information',
        'Pseudonyms are allowed',
        'The service will only respond to government requests that are reasonable',
        "You can access most of the pages on the service's website without revealing any personal information",
        'The service will resist legal requests for your information where reasonably possible',
        'Promises will be kept after a merger or acquisition',
        'Content is published under a free license instead of a bilateral one',
        'The service is not responsible for linked or (clearly) quoted content from third-party content providers',
        'You cannot delete your contributions, but it makes sense for this service',
        "This service honors 'Do Not Track' (DNT) headers",
        'Only aggregate data is given to third parties',
        'You are informed about the risk of publishing personal info online',
        'You must create an account to use this service',
        'You cannot distribute or disclose your account to third parties',
        'Usernames can be rejected or changed for any reason',
        'You can opt out of targeted advertising',
        'You can choose with whom you share content',
        'Third parties are not allowed to access your personal information without a legal basis',
        'An onion site accessible over Tor is provided',
        'Third parties are involved in operating the service',
        'A complaint mechanism is provided for the handling of personal data',
        'A free help desk is provided',
        'Minors must have the authorization of their legal guardians to use the service',
        'The service reviews its privacy policy on a regular basis',
        'Your content can be deleted if you violate the terms',
        "The copyright license that you grant this service is limited to the parties that make up the service's broader platform.",
        'You will be notified if personal data has been affected by data breaches',
        'Third parties used by the service are bound by confidentiality obligations',
        'Logs are deleted after a finite period of time',
        'This Service provides a list of Third Parties involved in its operation.',
        'The service promises to inform and/or notify you regarding government inquiries that may involve your personal data',
        'Separate policies are employed for different parts of the service',
        'You are responsible for any risks, damages, or losses that may incur by downloading materials',
        'You can delete your content from this service',
        "If you are the target of a copyright holder's take down notice, this service gives you the opportunity to defend yourself",
        'If you are the target of a copyright claim, your content may be removed',
        'This service provides a way for you to export your data',
        'Your use is throttled',
        'You can opt out of providing personal information to third parties',
        'Two factor authentication is provided for your account',
        "Per the service's terms, you may not express negative opinions about them",
        'This service has a no refund policy with some exceptions',
        'The publishing of personally identifiable information without the owner’s consent is not allowed',
        'You aren’t forced into binding arbitration in case of disputes',
        'This service does not collect, use, or share location data',
        'This service fines spammers',
        'The service does not guarantee that software errors will be corrected',
        'You are warned of the potential consequences related to third-party access',
        'You can choose the copyright license',
        'You are allowed to quote their content with attribution',
        'No third-party analytics or tracking platforms are used',
        'You are free to choose the type of copyright license that you want to use over your content',
        'The service explains how to prevent disclosure of personal information to third parties',
        'This service informs you that its Terms of Service does not apply to third party websites',
        'This service is only available for commercial use',
        'The service suspends your account after inactivity for a certain or uncertain time',
        'An anonymous payment method is offered',
        'Conditions may change, but your continued acceptance is not inferred from an earlier acceptance',
        'Tracking cookies refused will not limit your ability to use the service',
        'Only necessary logs are kept by the service to ensure quality',
        'Personal information that is shared outside its jurisdiction is processed according to the original jurisdiction’s data protection standards',
        'The service does not index or open files that you upload',
        "You can scrape the site, as long as it doesn't impact the server too much",
        'Only temporary session cookies are used',
        'Your personal data will not be used for an automated decision-making',
        'You aren’t allowed to remove or edit user-generated content',
        'This Service does not keep any logs.',
    ]
